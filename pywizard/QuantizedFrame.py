"""Index-based frame representation used by synthesis and editing."""

from __future__ import annotations

from dataclasses import dataclass

from lpcplayer.tables import tms_coeffs


@dataclass(frozen=True)
class QuantizedFrame:
    """A TMS52xx frame containing the indices written to the bitstream."""

    energy_index: int
    repeat_flag: bool = False
    pitch_index: int = 0
    k_indices: tuple[int, ...] = ()

    @property
    def is_silence(self) -> bool:
        return self.energy_index == 0

    @property
    def is_stop(self) -> bool:
        return self.energy_index == 15

    @property
    def is_voiced(self) -> bool:
        return not self.is_silence and not self.is_stop and self.pitch_index != 0

    def parameters(self) -> dict[str, int | bool]:
        """Return values in the format expected by the existing bit packer."""
        result: dict[str, int | bool] = {"kParameterGain": self.energy_index}
        if self.is_silence or self.is_stop:
            return result
        result["kParameterRepeat"] = self.repeat_flag
        result["kParameterPitch"] = self.pitch_index
        if not self.repeat_flag:
            count = 10 if self.is_voiced else 4
            for number, value in enumerate(self.k_indices[:count], 1):
                result[f"kParameterK{number}"] = value
        return result

    @classmethod
    def from_frame(cls, frame: object) -> "QuantizedFrame":
        """Freeze an existing analyzed frame at its quantized table indices."""
        parameters = frame.parameters()
        energy = int(parameters["kParameterGain"])
        if energy in (0, 15):
            return cls(energy_index=energy)
        repeat = bool(parameters["kParameterRepeat"])
        pitch = int(parameters["kParameterPitch"])
        count = 0 if repeat else (10 if pitch else 4)
        ks = tuple(int(parameters[f"kParameterK{i}"]) for i in range(1, count + 1))
        return cls(energy, repeat, pitch, ks)


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
