"""Perceptually mixed loss terms for analysis-by-synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pywizard.QuantizedFrame import QuantizedFrame


@dataclass(frozen=True)
class LossWeights:
    time: float
    preemphasis: float
    spectrum: float
    energy: float
    smoothness: float


PROFILES = {
    "waveform": LossWeights(0.60, 0.25, 0.05, 0.10, 0.005),
    "spectral": LossWeights(0.10, 0.15, 0.60, 0.15, 0.005),
    "balanced": LossWeights(0.25, 0.20, 0.40, 0.15, 0.005),
}


def _scaled_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    scale = np.sqrt(np.mean(reference * reference) / (np.mean(candidate * candidate) + 1e-12))
    return reference, candidate * min(scale, 20.0)


def _nmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean((reference - candidate) ** 2) / (np.mean(reference ** 2) + 1e-9))


def parameter_jump_penalty(frame: QuantizedFrame, neighbor: QuantizedFrame | None) -> float:
    """Return a small normalized discontinuity cost between speech frames."""
    if neighbor is None or frame.is_silence or frame.is_stop or neighbor.is_silence or neighbor.is_stop:
        return 0.0
    cost = abs(frame.energy_index - neighbor.energy_index) / 14.0
    cost += abs(frame.pitch_index - neighbor.pitch_index) / 63.0
    shared = min(len(frame.k_indices), len(neighbor.k_indices))
    if shared:
        cost += sum(abs(frame.k_indices[i] - neighbor.k_indices[i]) / 31.0 for i in range(shared)) / shared
    return cost


def perceptual_loss(
    reference: np.ndarray,
    candidate: np.ndarray,
    profile: str = "balanced",
    frame: QuantizedFrame | None = None,
    previous: QuantizedFrame | None = None,
    following: QuantizedFrame | None = None,
) -> float:
    """Combine waveform, pre-emphasis, spectrum, envelope, and smoothness errors."""
    weights = PROFILES[profile]
    reference, scaled = _scaled_pair(reference, candidate)
    time_error = _nmse(reference, scaled)
    ref_pre = np.concatenate(([reference[0]], reference[1:] - 0.97 * reference[:-1]))
    out_pre = np.concatenate(([scaled[0]], scaled[1:] - 0.97 * scaled[:-1]))
    pre_error = _nmse(ref_pre, out_pre)
    window = np.hanning(len(reference))
    ref_spectrum = np.log1p(np.abs(np.fft.rfft(reference * window)))
    out_spectrum = np.log1p(np.abs(np.fft.rfft(scaled * window)))
    spectrum_error = float(np.mean((ref_spectrum - out_spectrum) ** 2) / (np.mean(ref_spectrum ** 2) + 1e-9))
    block = 25
    ref_envelope = np.array([np.sqrt(np.mean(x * x) + 1e-12) for x in np.array_split(reference, max(1, len(reference) // block))])
    out_envelope = np.array([np.sqrt(np.mean(x * x) + 1e-12) for x in np.array_split(candidate, max(1, len(candidate) // block))])
    energy_error = _nmse(ref_envelope, out_envelope)
    smoothness = 0.0
    if frame is not None:
        smoothness = parameter_jump_penalty(frame, previous) + parameter_jump_penalty(frame, following)
    return (
        weights.time * time_error + weights.preemphasis * pre_error
        + weights.spectrum * spectrum_error + weights.energy * energy_error
        + weights.smoothness * smoothness
    )
