"""Deterministic offline TMS52xx digital speech synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from lpcplayer.tables import tms_coeffs
from pywizard.QuantizedFrame import QuantizedFrame


FRAME_SAMPLES = 200


@dataclass
class SynthesizerState:
    """Copyable interpolation, excitation, and lattice-filter state."""

    current: list[int] = field(default_factory=lambda: [0] * 12)
    target: list[int] = field(default_factory=lambda: [0] * 12)
    lattice: list[int] = field(default_factory=lambda: [0] * 10)
    period_counter: int = 0
    noise_lfsr: int = 0x1FFF
    initialized: bool = False

    def copy(self) -> "SynthesizerState":
        return SynthesizerState(
            self.current.copy(), self.target.copy(), self.lattice.copy(),
            self.period_counter, self.noise_lfsr, self.initialized,
        )


class TmsSynthesizer:
    """Synthesize indexed frames using TMS52xx tables and interpolation."""

    def __init__(self, tables: tms_coeffs):
        self.tables = tables

    def _set_target(self, frame: QuantizedFrame, state: SynthesizerState) -> None:
        if frame.is_stop:
            state.target[0] = 0
            return
        if frame.is_silence:
            state.target[0] = 0
            return
        state.target[0] = self.tables.energytable[frame.energy_index]
        state.target[1] = self.tables.pitchtable[frame.pitch_index]
        if not frame.repeat_flag:
            count = 10 if frame.pitch_index else 4
            for index in range(count):
                state.target[index + 2] = self.tables.ktable[index][frame.k_indices[index]]
            if not frame.pitch_index:
                state.target[6:12] = [0] * 6

    @staticmethod
    def _noise(state: SynthesizerState) -> int:
        bit = state.noise_lfsr & 1
        state.noise_lfsr = (state.noise_lfsr >> 1) ^ (0xB800 if bit else 0)
        return 1 if bit else -1

    def synthesize_frame(
        self, frame: QuantizedFrame, state: SynthesizerState | None = None
    ) -> tuple[np.ndarray, SynthesizerState]:
        """Generate one 200-sample frame, preserving state for later frames."""
        state = state.copy() if state is not None else SynthesizerState()
        self._set_target(frame, state)
        if not state.initialized:
            state.current[:] = state.target
            state.initialized = True

        output = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        for sample_number in range(FRAME_SAMPLES):
            if sample_number and sample_number % 25 == 0:
                period = sample_number // 25
                shift = self.tables.interp_coeff[period]
                for index in range(12):
                    state.current[index] += (
                        state.target[index] - state.current[index]
                    ) >> shift

            energy, pitch = state.current[:2]
            if energy == 0:
                excitation = 0
            elif pitch:
                state.period_counter = (state.period_counter + 1) % max(1, pitch)
                if state.period_counter < len(self.tables.chirptable):
                    chirp = self.tables.chirptable[state.period_counter]
                    chirp = chirp - 256 if chirp > 127 else chirp
                    excitation = (chirp * energy) >> 6
                else:
                    excitation = 0
            else:
                excitation = self._noise(state) * energy

            u = [0] * 11
            u[10] = excitation
            for order in range(9, -1, -1):
                u[order] = u[order + 1] - (
                    (state.current[order + 2] * state.lattice[order]) >> 9
                )
            u[0] = max(-512, min(511, u[0]))
            for order in range(9, 0, -1):
                state.lattice[order] = state.lattice[order - 1] + (
                    (state.current[order + 1] * u[order]) >> 9
                )
            state.lattice[0] = u[0]
            output[sample_number] = u[0] / 512.0
        return output, state

    def synthesize(self, frames: Iterable[QuantizedFrame]) -> np.ndarray:
        """Generate an utterance from reset state, stopping at a stop frame."""
        state = SynthesizerState()
        chunks = []
        for frame in frames:
            if frame.is_stop:
                break
            chunk, state = self.synthesize_frame(frame, state)
            chunks.append(chunk)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
