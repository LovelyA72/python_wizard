"""Index-based frame representation used by synthesis and optimization."""

from __future__ import annotations

from dataclasses import dataclass


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

    def as_dict(self) -> dict[str, object]:
        return {
            "energy_index": self.energy_index,
            "repeat_flag": self.repeat_flag,
            "pitch_index": self.pitch_index,
            "k_indices": list(self.k_indices),
        }

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

