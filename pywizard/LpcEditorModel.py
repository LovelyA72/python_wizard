"""Quantized LPC editing, independent of Tk and audio devices."""

from dataclasses import replace
import math

from lpcplayer.tables import tables
from pywizard.QuantizedFrame import QuantizedFrame


class LpcEditorModel:
    def __init__(self, frames, chip_variant):
        self.tables = tables[chip_variant]
        self.chip_variant = chip_variant
        if chip_variant.lower() == "talkie":
            from pywizard.TalkieSynthesizer import K_TABLES
            self._coefficient_tables = tuple(
                tuple(value / (32768 if i < 2 else 128) for value in values)
                for i, values in enumerate(K_TABLES)
            )
        else:
            self._coefficient_tables = tuple(
                tuple(value / 512 for value in values) for values in self.tables.ktable
            )
        self._neutral = tuple(
            min(range(len(table)), key=lambda n: abs(table[n]))
            for table in self._coefficient_tables
        )
        self._frames = []
        self.has_stop = False
        for frame in frames:
            frame = QuantizedFrame.from_frame(frame)
            if frame.is_stop:
                self.has_stop = True
                break
            self._frames.append(frame)
        if not self._frames:
            raise ValueError("No speech frames to edit")
        self.selection = (0, 0)
        self.undo_stack = []
        self.redo_stack = []
        self._before = None

    @property
    def count(self):
        return len(self._frames)

    @property
    def frames(self):
        return self._frames.copy() + ([QuantizedFrame(15)] if self.has_stop else [])

    @property
    def pitch_table(self):
        if self.chip_variant.lower() == "talkie":
            from pywizard.TalkieSynthesizer import PERIOD
            return PERIOD
        return self.tables.pitchtable

    def coefficient_table(self, index):
        return self._coefficient_tables[index]

    def select(self, first, last):
        first = max(0, min(self.count - 1, int(first)))
        last = max(0, min(self.count - 1, int(last)))
        self.selection = tuple(sorted((first, last)))

    def _expanded(self):
        # Freeze effective coefficients before editing: a following repeat must
        # not accidentally inherit the newly drawn coefficients of its neighbor.
        ks = list(self._neutral)
        frames = []
        for frame in self._frames:
            if frame.is_silence:
                frames.append(frame)
                continue
            count = 10 if frame.is_voiced else 4
            if not frame.repeat_flag:
                ks[:count] = frame.k_indices
                if not frame.is_voiced:
                    ks[4:] = self._neutral[4:]
            frames.append(replace(frame, repeat_flag=False, k_indices=tuple(ks[:count])))
        return frames

    def values(self, track):
        frames = self._expanded()
        if track == "gain":
            return [frame.energy_index for frame in frames]
        if track == "pitch":
            return [8000 / self.pitch_table[f.pitch_index] if f.is_voiced else 0 for f in frames]
        k = int(track[1:]) - 1
        table = self.coefficient_table(k)
        return [table[f.k_indices[k]] if k < len(f.k_indices) else None for f in frames]

    def begin(self):
        if self._before is None:
            self._before = self._frames.copy()

    def commit(self):
        if self._before is None:
            return False
        before, self._before = self._before, None
        if before == self._frames:
            return False
        self.undo_stack.append(before)
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()
        return True

    def set_value(self, index, track, value):
        if not math.isfinite(value):
            raise ValueError("Value must be finite")
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.begin()
        frames = self._expanded()
        frame = frames[index]
        if track == "gain":
            energy = max(0, min(14, round(value)))
            if energy == 0:
                frame = QuantizedFrame(0)
            elif frame.is_silence:
                neutral = self._neutral[:4]
                frame = QuantizedFrame(energy, False, 0, neutral)
            else:
                frame = replace(frame, energy_index=energy)
        elif track == "pitch":
            if frame.is_silence:
                return
            pitch = 0 if value <= 0 else min(
                range(1, len(self.pitch_table)),
                key=lambda n: abs(8000 / self.pitch_table[n] - value),
            )
            ks = list(frame.k_indices)
            for i in range(len(ks), 10 if pitch else 4):
                ks.append(self._neutral[i])
            frame = replace(frame, pitch_index=pitch, k_indices=tuple(ks[:10 if pitch else 4]))
        else:
            k = int(track[1:]) - 1
            if not 0 <= k < 10:
                raise ValueError("Unsupported coefficient")
            if frame.is_silence or k >= len(frame.k_indices):
                return
            ks = list(frame.k_indices)
            table = self.coefficient_table(k)
            ks[k] = min(range(len(table)), key=lambda n: abs(table[n] - value))
            frame = replace(frame, k_indices=tuple(ks))
        if frame == frames[index]:
            return
        frames[index] = frame
        self._frames = frames

    def set_selection(self, track, value):
        self.begin()
        first, last = self.selection
        for index in range(first, last + 1):
            self.set_value(index, track, value)
        return self.commit()

    def undo(self):
        self.commit()
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._frames.copy())
        self._frames = self.undo_stack.pop()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._frames.copy())
        self._frames = self.redo_stack.pop()
        return True
