"""Source-relative Talkie burst suppression and fixed-point stabilization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Sequence

import numpy as np
from scipy.signal import resample

from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TalkieSynthesizer import (
    FRAME_SAMPLES,
    K_TABLES,
    TalkieState,
    TalkieSynthesizer,
)


@dataclass(frozen=True)
class TalkieMitigationConfig:
    """Conservative source-relative transient limits."""

    strength: float = 1.0
    gate_threshold: float = 0.0
    gate_frames: int = 3
    rms_ratio: float = 2.25
    preemphasis_ratio: float = 2.25
    jump_ratio: float = 2.0
    peak_ratio: float = 2.0
    rms_floor: float = 0.03
    preemphasis_floor: float = 0.05
    jump_floor: float = 0.25
    peak_floor: float = 0.25
    input_frame_samples: int = FRAME_SAMPLES
    protect_voiced: bool = False


@dataclass
class TalkieMitigationResult:
    frames: list[QuantizedFrame]
    changes: list[dict[str, object]]
    candidate_evaluations: int
    elapsed_seconds: float

    @property
    def changed_frames(self) -> int:
        return len(self.changes)

    @property
    def gated_frames(self) -> int:
        return sum(
            change.get("kind") in ("gate", "gate_transition")
            for change in self.changes
        )


def _copy_state(state: TalkieState) -> TalkieState:
    return state.copy()


def _render_frame(
    frame: QuantizedFrame, state: TalkieState,
) -> tuple[np.ndarray, TalkieState]:
    """Render through the unchanged Talkie preview implementation."""
    return TalkieSynthesizer().synthesize_frame(frame, state)


def _target_frames(
    samples: np.ndarray, count: int, input_frame_samples: int,
) -> list[np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    targets = []
    for index in range(count):
        start = index * input_frame_samples
        chunk = samples[start:start + input_frame_samples]
        chunk = np.pad(chunk, (0, input_frame_samples - len(chunk)))
        if input_frame_samples != FRAME_SAMPLES:
            chunk = resample(chunk, FRAME_SAMPLES)
        targets.append(np.asarray(chunk, dtype=np.float64))
    return targets


def _metrics(samples: np.ndarray) -> tuple[float, float, float, float]:
    rms = float(np.sqrt(np.mean(samples * samples)))
    preemphasized = samples[1:] - 0.97 * samples[:-1]
    preemphasis_rms = float(np.sqrt(np.mean(preemphasized * preemphasized)))
    maximum_jump = float(np.max(np.abs(np.diff(samples))))
    peak = float(np.max(np.abs(samples)))
    return rms, preemphasis_rms, maximum_jump, peak


def _burst_risk(
    audio: np.ndarray, reference: np.ndarray, config: TalkieMitigationConfig,
) -> tuple[float, dict[str, float]]:
    rms, preemphasis, jump, peak = _metrics(audio)
    ref_rms, ref_preemphasis, ref_jump, ref_peak = _metrics(reference)
    limits = {
        "rms": max(config.rms_floor, config.rms_ratio * ref_rms),
        "preemphasis": max(
            config.preemphasis_floor,
            config.preemphasis_ratio * ref_preemphasis,
        ),
        "jump": max(config.jump_floor, config.jump_ratio * ref_jump),
        "peak": max(config.peak_floor, config.peak_ratio * ref_peak),
    }
    measured = {
        "rms": rms,
        "preemphasis": preemphasis,
        "jump": jump,
        "peak": peak,
    }
    risk = config.strength * max(
        measured[name] / limits[name] for name in measured
    )
    return risk, measured


def silence_transition_frame() -> QuantizedFrame:
    """Return a zero-excitation frame that flushes Talkie's lattice filter.

    A rest frame only clears Talkie's energy and retains its filter state.
    Minimum Talkie energy combined with voiced excitation rounds every chirp
    sample to zero, while a nonzero pitch makes the frame load all ten
    coefficients.  Choosing the smallest nonnegative coefficient in each
    table avoids the negative fixed-point limit cycle produced by choosing
    the closest-to-zero entries.
    """
    neutral_coefficients = tuple(
        next(index for index, value in enumerate(table) if value >= 0)
        for table in K_TABLES
    )
    return QuantizedFrame(
        energy_index=1,
        repeat_flag=False,
        pitch_index=1,
        k_indices=neutral_coefficients,
    )


def _apply_noise_gate(
    frames: list[QuantizedFrame],
    targets: Sequence[np.ndarray],
    config: TalkieMitigationConfig,
) -> tuple[
    list[QuantizedFrame],
    list[dict[str, object]],
    list[tuple[int, ...]],
]:
    """Silence complete source-RMS runs that satisfy the gate hold time."""
    output = frames.copy()
    changes: list[dict[str, object]] = []
    run: list[tuple[int, float]] = []
    gated_runs: list[tuple[int, ...]] = []
    target_index = 0

    def flush_run() -> None:
        if len(run) >= config.gate_frames:
            gated_runs.append(tuple(frame_index for frame_index, _ in run))
            for frame_index, source_rms in run:
                old = output[frame_index]
                if old.is_silence:
                    continue
                output[frame_index] = QuantizedFrame(0)
                changes.append({
                    "kind": "gate",
                    "frame": frame_index,
                    "source_rms": source_rms,
                    "threshold": config.gate_threshold,
                    "old_energy_index": old.energy_index,
                    "new_energy_index": 0,
                })
        run.clear()

    for frame_index, frame in enumerate(output):
        if frame.is_stop:
            break
        source_rms = float(np.sqrt(np.mean(targets[target_index] ** 2)))
        target_index += 1
        if source_rms < config.gate_threshold:
            run.append((frame_index, source_rms))
        else:
            flush_run()
    flush_run()
    return output, changes, gated_runs


def _stabilize_gated_silence(
    frames: list[QuantizedFrame],
    changes: list[dict[str, object]],
    gated_runs: Sequence[tuple[int, ...]],
) -> tuple[list[QuantizedFrame], list[dict[str, object]]]:
    """Flush retained Talkie state when a gated run sustains a limit cycle."""
    output = frames.copy()
    changes_by_frame = {
        int(change["frame"]): change
        for change in changes
        if change.get("kind") == "gate"
    }
    runs_by_start = {run[0]: run for run in gated_runs if run}
    state = TalkieState()
    index = 0

    while index < len(output):
        frame = output[index]
        if frame.is_stop:
            break

        run = runs_by_start.get(index)
        if run is None:
            _, state = _render_frame(frame, state)
            index += 1
            continue

        trial_state = _copy_state(state)
        last_audio = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        for frame_index in run:
            last_audio, trial_state = _render_frame(
                output[frame_index],
                trial_state,
            )

        if np.any(last_audio):
            old = output[index]
            transition = silence_transition_frame()
            output[index] = transition
            change = changes_by_frame.get(index)
            if change is None:
                change = {
                    "frame": index,
                    "source_rms": 0.0,
                    "threshold": None,
                    "old_energy_index": old.energy_index,
                }
                changes.append(change)
            change.update({
                "kind": "gate_transition",
                "new_energy_index": transition.energy_index,
                "new_pitch_index": transition.pitch_index,
                "new_k_indices": transition.k_indices,
            })

        for frame_index in run:
            _, state = _render_frame(output[frame_index], state)
        index = run[-1] + 1

    return output, changes


def mitigate_talkie_frames(
    frames: Sequence[QuantizedFrame],
    original_pcm: np.ndarray,
    config: TalkieMitigationConfig = TalkieMitigationConfig(),
) -> TalkieMitigationResult:
    """Limit disproportionate Talkie output relative to source PCM.

    The optional source gate emits rest frames. If Talkie's retained lattice
    state would sustain a fixed-point limit cycle through a gated run, its
    first frame becomes a zero-excitation transition that loads stable
    coefficients before the remaining rest frames. The burst guard otherwise
    chooses the highest lower legal energy index that brings a flagged
    unvoiced frame within source-relative transient limits. ``protect_voiced``
    also checks periodic frames and permits coefficient damping if reduced
    gain cannot stop a retained-state burst. QBP enables this option.
    """
    started = time.perf_counter()
    output = list(frames)
    if not 0.0 <= config.strength <= 3.0:
        raise ValueError("Talkie mitigation strength must be between 0 and 3")
    if not 0.0 <= config.gate_threshold <= 1.0:
        raise ValueError("Talkie gate threshold must be between 0 and 1")
    if config.gate_frames < 1:
        raise ValueError("Talkie gate hold must be at least one frame")
    speech_count = sum(not frame.is_stop for frame in output)
    targets = _target_frames(
        original_pcm, speech_count, max(1, config.input_frame_samples),
    )
    changes: list[dict[str, object]] = []
    gated_runs: list[tuple[int, ...]] = []
    if config.gate_threshold > 0.0:
        output, changes, gated_runs = _apply_noise_gate(
            output, targets, config,
        )
    if config.strength <= 0.0:
        output, changes = _stabilize_gated_silence(
            output, changes, gated_runs,
        )
        return TalkieMitigationResult(
            frames=output,
            changes=changes,
            candidate_evaluations=0,
            elapsed_seconds=time.perf_counter() - started,
        )
    state = TalkieState()
    evaluations = 0
    target_index = 0

    for index, frame in enumerate(output):
        if frame.is_stop:
            break
        reference = targets[target_index]
        target_index += 1
        audio, next_state = _render_frame(frame, state)
        initial_risk, initial_metrics = _burst_risk(audio, reference, config)

        # Sequence encoders can also overdrive periodic excitation. Keep the
        # historical noise-only guard unless the caller opts into both types.
        if (frame.pitch_index == 0 or config.protect_voiced) and not frame.is_silence and initial_risk > 1.0:
            selected = frame
            selected_audio = audio
            selected_state = next_state
            selected_risk = initial_risk
            selected_metrics = initial_metrics

            for energy_index in range(frame.energy_index - 1, 0, -1):
                candidate = replace(frame, energy_index=energy_index)
                candidate_audio, candidate_state = _render_frame(candidate, state)
                candidate_risk, candidate_metrics = _burst_risk(
                    candidate_audio, reference, config,
                )
                evaluations += 1
                if candidate_risk < selected_risk:
                    selected = candidate
                    selected_audio = candidate_audio
                    selected_state = candidate_state
                    selected_risk = candidate_risk
                    selected_metrics = candidate_metrics
                if candidate_risk <= 1.0:
                    break

            # A strongly resonant lattice can keep buzzing even at minimum
            # excitation. QBP opts into coefficient damping when gain alone
            # cannot control it. Re-render from the same incoming state so
            # every candidate includes the actual fixed-point history.
            stabilized = False
            if config.protect_voiced and selected_risk > 1.0 and not frame.repeat_flag:
                for scale in (0.95, 0.85, 0.65, 0.4, 0.0):
                    ks = tuple(
                        (next(v for v, value in enumerate(K_TABLES[k]) if value >= 0)
                         if scale == 0 else
                         min(range(len(K_TABLES[k])),
                             key=lambda v: abs(K_TABLES[k][v] - K_TABLES[k][old] * scale)))
                        for k, old in enumerate(frame.k_indices)
                    )
                    candidate = replace(selected, k_indices=ks)
                    candidate_audio, candidate_state = _render_frame(candidate, state)
                    candidate_risk, candidate_metrics = _burst_risk(candidate_audio, reference, config)
                    evaluations += 1
                    if candidate_risk < selected_risk:
                        selected, selected_audio, selected_state = candidate, candidate_audio, candidate_state
                        selected_risk, selected_metrics = candidate_risk, candidate_metrics
                        stabilized = True
                    if candidate_risk <= 1.0:
                        break

            if selected != frame:
                output[index] = selected
                audio, next_state = selected_audio, selected_state
                changes.append({
                    "kind": "stabilize" if stabilized else "burst",
                    "frame": index,
                    "old_energy_index": frame.energy_index,
                    "new_energy_index": selected.energy_index,
                    "old_risk": initial_risk,
                    "new_risk": selected_risk,
                    "old_metrics": initial_metrics,
                    "new_metrics": selected_metrics,
                    "old_k_indices": frame.k_indices,
                    "new_k_indices": selected.k_indices,
                })

        state = next_state

    output, changes = _stabilize_gated_silence(
        output, changes, gated_runs,
    )
    return TalkieMitigationResult(
        frames=output,
        changes=changes,
        candidate_evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - started,
    )
