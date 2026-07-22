"""Offline port of the Arduino Talkie 1.4.0 preview synthesizer.

The encoder still uses the TMS5220 frame layout and quantization tables.  This
module reproduces Talkie's default (non-FAST_8BIT_MODE, non-10-bit) playback
path: its software coefficient tables, excitation, lattice filter, and 8-bit
PWM conversion.  The source ported here is ``Talkie/src/Talkie.cpp`` and
``Talkie/src/TalkieLPC.h``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from pywizard.QuantizedFrame import QuantizedFrame


FRAME_SAMPLES = 200

ENERGY = (0, 2, 3, 4, 5, 7, 10, 15, 20, 32, 41, 57, 81, 114, 161, 255)
PERIOD = (
    0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47, 49,
    51, 53, 54, 57, 59, 61, 63, 66, 69, 71, 73, 77, 79, 81, 85, 87,
    92, 95, 99, 102, 106, 110, 115, 119, 123, 128, 133, 138, 143, 149,
    154, 160,
)
K_TABLES = (
    (-32064, -31872, -31808, -31680, -31552, -31424, -31232, -30848,
     -30592, -30336, -30016, -29696, -29376, -28928, -28480, -27968,
     -26368, -24256, -21632, -18368, -14528, -10048, -5184, 0,
     5184, 10048, 14528, 18368, 21632, 24256, 26368, 27968),
    (-20992, -19328, -17536, -15552, -13440, -11200, -8768, -6272,
     -3712, -1088, 1536, 4160, 6720, 9216, 11584, 13824,
     15936, 17856, 19648, 21248, 22656, 24000, 25152, 26176,
     27072, 27840, 28544, 29120, 29632, 30080, 30464, 32384),
    (-110, -97, -83, -70, -56, -43, -29, -16, -2, 11, 25, 38, 52, 65, 79, 92),
    (-82, -68, -54, -40, -26, -12, 1, 15, 29, 43, 57, 71, 85, 99, 113, 126),
    (-82, -70, -59, -47, -35, -24, -12, -1, 11, 23, 34, 46, 57, 69, 81, 92),
    (-64, -53, -42, -31, -20, -9, 3, 14, 25, 36, 47, 58, 69, 80, 91, 102),
    (-77, -65, -53, -41, -29, -17, -5, 7, 19, 31, 43, 55, 67, 79, 90, 102),
    (-64, -40, -16, 7, 31, 55, 79, 102),
    (-64, -44, -24, -4, 16, 37, 57, 77),
    (-51, -33, -15, 4, 22, 32, 59, 77),
)
CHIRP = (0, 3, 15, 40, 76, 108, 113, 80, 37, 38, 76, 68, 26, 50, 59, 19, 55, 26, 37, 31, 29)


def _int16(value: int) -> int:
    """Apply the signed 16-bit storage used by Talkie's default path."""
    return ((value + 0x8000) & 0xFFFF) - 0x8000


@dataclass
class TalkieState:
    energy: int = 0
    period: int = 0
    coefficients: list[int] = field(default_factory=lambda: [0] * 10)
    lattice: list[int] = field(default_factory=lambda: [0] * 10)
    period_counter: int = 0
    noise_lfsr: int = 1


class _BitReader:
    def __init__(self, data: bytes | bytearray | Iterable[int]):
        self.data = bytes(data)
        self.position = 0

    def read(self, count: int) -> int:
        if self.position + count > len(self.data) * 8:
            raise EOFError
        value = 0
        for _ in range(count):
            value = (value << 1) | (
                (self.data[self.position // 8] >> (self.position % 8)) & 1
            )
            self.position += 1
        return value


class TalkieSynthesizer:
    """Synthesize TMS5220-format indexed frames as Arduino Talkie does."""

    @staticmethod
    def decode_frames(data: bytes | bytearray | Iterable[int]) -> list[QuantizedFrame]:
        """Decode the LSB-first Talkie/TMS5220 byte stream into indexed frames."""
        reader = _BitReader(data)
        frames: list[QuantizedFrame] = []
        try:
            while True:
                energy = reader.read(4)
                if energy in (0, 15):
                    frame = QuantizedFrame(energy)
                else:
                    repeat = bool(reader.read(1))
                    pitch = reader.read(6)
                    count = 0 if repeat else (10 if pitch else 4)
                    widths = (5, 5, 4, 4, 4, 4, 4, 3, 3, 3)
                    indices = tuple(reader.read(widths[i]) for i in range(count))
                    frame = QuantizedFrame(energy, repeat, pitch, indices)
                frames.append(frame)
                if frame.is_stop:
                    break
        except EOFError:
            pass
        return frames

    @staticmethod
    def _load_frame(frame: QuantizedFrame, state: TalkieState) -> None:
        if frame.is_silence:
            state.energy = 0
            return
        state.energy = ENERGY[frame.energy_index]
        state.period = PERIOD[frame.pitch_index]
        if not frame.repeat_flag:
            count = 10 if frame.pitch_index else 4
            for index in range(count):
                state.coefficients[index] = K_TABLES[index][frame.k_indices[index]]

    @staticmethod
    def _sample(state: TalkieState) -> int:
        if state.period:
            if state.period_counter < state.period:
                state.period_counter += 1
            else:
                state.period_counter = 0
            excitation = (
                (CHIRP[state.period_counter] * state.energy) >> 8
                if state.period_counter < len(CHIRP) else 0
            )
        else:
            state.noise_lfsr = (
                (state.noise_lfsr >> 1)
                ^ (0xB800 if state.noise_lfsr & 1 else 0)
            ) & 0xFFFF
            excitation = state.energy if state.noise_lfsr & 1 else -state.energy

        u = [0] * 11
        u[10] = excitation
        # K3..K10 are signed 8-bit coefficients with seven fractional bits.
        for order in range(9, 1, -1):
            u[order] = _int16(
                u[order + 1]
                - ((state.coefficients[order] * state.lattice[order]) >> 7)
            )
        # Default Talkie keeps K1/K2 as signed Q15 values.
        for order in (1, 0):
            u[order] = _int16(
                u[order + 1]
                - ((state.coefficients[order] * state.lattice[order]) >> 15)
            )

        for order in range(9, 2, -1):
            state.lattice[order] = _int16(
                state.lattice[order - 1]
                + ((state.coefficients[order - 1] * u[order - 1]) >> 7)
            )
        for order in (2, 1):
            coefficient = state.coefficients[order - 1]
            state.lattice[order] = _int16(
                state.lattice[order - 1]
                + ((coefficient * u[order - 1]) >> 15)
            )
        state.lattice[0] = u[0]
        return ((u[0] >> 2) + 0x80) & 0xFF

    def synthesize_u8(self, frames: Iterable[QuantizedFrame]) -> np.ndarray:
        """Return Talkie's unsigned 8-bit PWM samples."""
        state = TalkieState()
        output: list[int] = []
        for frame in frames:
            if frame.is_stop:
                break
            self._load_frame(frame, state)
            output.extend(self._sample(state) for _ in range(FRAME_SAMPLES))
        return np.asarray(output, dtype=np.uint8)

    def synthesize(self, frames: Iterable[QuantizedFrame]) -> np.ndarray:
        """Return centered floating-point samples for WAV export."""
        samples = self.synthesize_u8(frames)
        return (samples.astype(np.float64) - 128.0) / 128.0

    def decode(self, data: bytes | bytearray | Iterable[int]) -> list[int]:
        """Decode a packed LPC byte stream to samples suitable for PyAudio."""
        return self.synthesize_u8(self.decode_frames(data)).tolist()
