"""Bounded coordinate-descent optimization of quantized TMS52xx frames."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from lpcplayer.tables import tms_coeffs
from pywizard.OptimizationLoss import perceptual_loss
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TmsSynthesizer import FRAME_SAMPLES, SynthesizerState, TmsSynthesizer


@dataclass(frozen=True)
class OptimizationConfig:
    """Conservative controls for the local candidate search."""

    passes: int = 2
    radius: int = 1
    lookahead: int = 1
    loss_profile: str = "balanced"
    input_frame_samples: int = FRAME_SAMPLES


@dataclass
class OptimizationResult:
    frames: list[QuantizedFrame]
    initial_loss: float
    optimized_loss: float
    candidate_evaluations: int
    elapsed_seconds: float
    changes: list[dict[str, object]]
    accepted_loss_pairs: list[tuple[float, float]]

    @property
    def changed_frames(self) -> int:
        return len(self.changes)

    @property
    def improvement_percent(self) -> float:
        if self.initial_loss <= 0:
            return 0.0
        return 100.0 * (self.initial_loss - self.optimized_loss) / self.initial_loss


def neighbor_indices(value: int, size: int, radius: int) -> tuple[int, ...]:
    """Generate unique nearby table indices, clamped at table boundaries."""
    return tuple(sorted({max(0, min(size - 1, value + offset)) for offset in range(-radius, radius + 1)}))


def validate_frame(frame: QuantizedFrame, tables: tms_coeffs) -> bool:
    """Reject frames that cannot be represented by the selected chip tables."""
    if not 0 <= frame.energy_index < len(tables.energytable):
        return False
    if frame.is_silence or frame.is_stop:
        return frame.pitch_index == 0 and not frame.k_indices
    if not 0 <= frame.pitch_index < len(tables.pitchtable):
        return False
    if frame.repeat_flag:
        return not frame.k_indices
    count = 10 if frame.pitch_index else 4
    return len(frame.k_indices) == count and all(
        0 <= value < len(tables.ktable[index])
        for index, value in enumerate(frame.k_indices)
    )


def _targets(samples: np.ndarray, count: int, input_frame_samples: int) -> list[np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    result = []
    for index in range(count):
        start = index * input_frame_samples
        chunk = samples[start:start + input_frame_samples]
        chunk = np.pad(chunk, (0, input_frame_samples - len(chunk)))
        if input_frame_samples != FRAME_SAMPLES:
            chunk = resample(chunk, FRAME_SAMPLES)
        result.append(np.asarray(chunk, dtype=np.float64))
    return result


def _sequence_losses(
    frames: Sequence[QuantizedFrame], targets: Sequence[np.ndarray], synth: TmsSynthesizer,
    profile: str,
) -> tuple[float, list[float]]:
    state = SynthesizerState()
    losses = []
    speech_index = 0
    for index, frame in enumerate(frames):
        if frame.is_stop:
            break
        audio, state = synth.synthesize_frame(frame, state)
        previous = frames[index - 1] if index else None
        following = frames[index + 1] if index + 1 < len(frames) else None
        losses.append(perceptual_loss(targets[speech_index], audio, profile, frame, previous, following))
        speech_index += 1
    return float(sum(losses)), losses


def _candidate_loss(
    candidate: QuantizedFrame, index: int, frames: Sequence[QuantizedFrame],
    targets: Sequence[np.ndarray], state_before: SynthesizerState,
    synth: TmsSynthesizer, config: OptimizationConfig,
) -> float:
    audio, state = synth.synthesize_frame(candidate, state_before)
    previous = frames[index - 1] if index else None
    following = frames[index + 1] if index + 1 < len(frames) else None
    loss = perceptual_loss(targets[index], audio, config.loss_profile, candidate, previous, following)
    if config.lookahead and following is not None and not following.is_stop:
        next_audio, _ = synth.synthesize_frame(following, state)
        boundary = min(25 * config.lookahead, FRAME_SAMPLES)
        loss += 0.1 * perceptual_loss(
            targets[index + 1][:boundary], next_audio[:boundary], config.loss_profile
        )
    return loss


def optimize_frames(
    frames: Sequence[QuantizedFrame], original_pcm: np.ndarray, tables: tms_coeffs,
    config: OptimizationConfig = OptimizationConfig(),
    progress_callback: Callable[[int, int], None] | None = None,
) -> OptimizationResult:
    """Optimize actual table indices without altering stream structure or flags."""
    started = time.perf_counter()
    initial = list(frames)
    optimized = list(frames)
    speech_count = sum(not frame.is_stop for frame in optimized)
    targets = _targets(original_pcm, speech_count, max(1, config.input_frame_samples))
    synth = TmsSynthesizer(tables)
    initial_loss, initial_frame_losses = _sequence_losses(initial, targets, synth, config.loss_profile)
    evaluations = 0
    accepted_loss_pairs: list[tuple[float, float]] = []
    progress_per_pass = sum(
        1 if frame.is_silence else 2 + (0 if frame.repeat_flag else len(frame.k_indices))
        for frame in optimized if not frame.is_stop
    )
    total_progress = max(1, config.passes) * max(1, progress_per_pass)
    completed_progress = 0
    if progress_callback is not None:
        progress_callback(0, total_progress)

    for _ in range(max(0, config.passes)):
        state = SynthesizerState()
        improved_in_pass = False
        for index, frame in enumerate(optimized):
            if frame.is_stop:
                break
            if frame.is_silence:
                _, state = synth.synthesize_frame(frame, state)
                completed_progress += 1
                if progress_callback is not None:
                    progress_callback(completed_progress, total_progress)
                continue

            current = frame
            coordinates: list[tuple[str, int | None, int]] = [
                ("energy_index", None, len(tables.energytable)),
                ("pitch_index", None, len(tables.pitchtable)),
            ]
            if not current.repeat_flag:
                coordinates.extend(
                    ("k_indices", k, len(tables.ktable[k])) for k in range(len(current.k_indices))
                )

            for name, position, size in coordinates:
                value = getattr(current, name) if position is None else current.k_indices[position]
                values = neighbor_indices(int(value), size, max(0, config.radius))
                # Preserve speech/silence and voiced/unvoiced structure in the first version.
                if name == "energy_index":
                    values = tuple(v for v in values if 1 <= v <= 14)
                elif name == "pitch_index":
                    values = tuple(v for v in values if (v == 0) == (current.pitch_index == 0))

                best = current
                best_loss = _candidate_loss(current, index, optimized, targets, state, synth, config)
                for value in values:
                    if name == "k_indices":
                        ks = list(current.k_indices)
                        ks[int(position)] = value
                        candidate = replace(current, k_indices=tuple(ks))
                    else:
                        candidate = replace(current, **{name: value})
                    if candidate == current or not validate_frame(candidate, tables):
                        continue
                    candidate_loss = _candidate_loss(candidate, index, optimized, targets, state, synth, config)
                    evaluations += 1
                    if candidate_loss + 1e-12 < best_loss:
                        best, best_loss = candidate, candidate_loss
                if best != current:
                    previous_loss = _candidate_loss(current, index, optimized, targets, state, synth, config)
                    accepted_loss_pairs.append((previous_loss, best_loss))
                    current = best
                    optimized[index] = current
                    improved_in_pass = True
                completed_progress += 1
                if progress_callback is not None:
                    progress_callback(completed_progress, total_progress)
            _, state = synth.synthesize_frame(current, state)
        if not improved_in_pass:
            break

    optimized_loss, optimized_frame_losses = _sequence_losses(optimized, targets, synth, config.loss_profile)
    # Local context is an approximation. Never publish a sequence whose full
    # objective is worse than the initial quantization.
    if optimized_loss > initial_loss + 1e-12:
        optimized = initial.copy()
        optimized_loss = initial_loss
        optimized_frame_losses = initial_frame_losses.copy()
    if progress_callback is not None:
        progress_callback(total_progress, total_progress)
    changes = []
    for index, (old, new) in enumerate(zip(initial, optimized)):
        if old == new:
            continue
        changed = []
        if old.energy_index != new.energy_index:
            changed.append("energy")
        if old.pitch_index != new.pitch_index:
            changed.append("pitch")
        changed.extend(
            f"K{k + 1}" for k, (a, b) in enumerate(zip(old.k_indices, new.k_indices)) if a != b
        )
        changes.append({
            "frame": index, "old_indices": old.as_dict(), "new_indices": new.as_dict(),
            "old_loss": initial_frame_losses[index], "new_loss": optimized_frame_losses[index],
            "parameters_changed": changed,
        })
    return OptimizationResult(
        optimized, initial_loss, optimized_loss, evaluations,
        time.perf_counter() - started, changes, accepted_loss_pairs,
    )


def write_synthesized_wav(path: str | Path, audio: np.ndarray) -> None:
    """Write normalized 8 kHz, signed 16-bit comparison audio."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak:
        audio = audio / max(1.0, peak)
    wavfile.write(str(path), 8000, np.asarray(np.clip(audio, -1, 1) * 32767, dtype=np.int16))


def write_report(path: str | Path, result: OptimizationResult) -> None:
    data = {
        "frame_count": len(result.frames), "changed_frames": result.changed_frames,
        "initial_loss": result.initial_loss, "optimized_loss": result.optimized_loss,
        "improvement_percent": result.improvement_percent,
        "encoding_seconds": result.elapsed_seconds,
        "candidate_evaluations": result.candidate_evaluations,
        "parameter_change_counts": dict(Counter(p for c in result.changes for p in c["parameters_changed"])),
        "changes": result.changes,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
