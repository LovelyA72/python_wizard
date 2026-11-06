import unittest

import numpy as np

from pywizard.CodingTable import CodingTable
from pywizard.FrameDataBinaryEncoder import BitPacker
from pywizard.FrameOptimizer import validate_frame
from pywizard.GuiConversion import ConvertedProcessor
from pywizard.LpcEditorModel import LpcEditorModel
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TalkieSynthesizer import TalkieSynthesizer
from lpcplayer.tables import tables


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.first = QuantizedFrame(8, False, 30, (10, 12, 7, 7, 7, 7, 7, 3, 3, 3))
        self.repeat = QuantizedFrame(7, True, 30)
        self.frames = [self.first, self.repeat, QuantizedFrame(15)]
        self.model = LpcEditorModel(self.frames, "tms5220")

    def test_stop_excluded_from_selection_and_preserved_in_stream(self):
        self.model.select(-10, 100)
        self.assertEqual(self.model.selection, (0, 1))
        self.model.set_selection("gain", 99)
        self.assertEqual([f.energy_index for f in self.model.frames], [14, 14, 15])
        processor = ConvertedProcessor(self.model.frames, CodingTable("tms5220"))
        self.assertEqual(TalkieSynthesizer.decode_frames(BitPacker.lpc_bytes(processor)), self.model.frames)

    def test_edit_does_not_change_later_repeat_coefficients(self):
        self.model.set_selection("k1", .9)
        changed = self.model.frames
        self.assertNotEqual(changed[0].k_indices[0], self.first.k_indices[0])
        self.assertEqual(changed[1].k_indices, self.first.k_indices)
        self.assertFalse(changed[1].repeat_flag)
        self.assertEqual(self.frames[0], self.first)

    def test_gain_only_expansion_preserves_repeat_synthesis(self):
        original = TalkieSynthesizer().synthesize(self.frames)
        expanded = TalkieSynthesizer().synthesize(self.model._expanded() + [QuantizedFrame(15)])
        np.testing.assert_array_equal(original, expanded)

    def test_undo_redo_restore_original_repeat_flags(self):
        self.model.set_selection("gain", 12)
        edited = self.model.frames
        self.assertTrue(self.model.undo())
        self.assertEqual(self.model.frames, self.frames)
        self.assertTrue(self.model.redo())
        self.assertEqual(self.model.frames, edited)
        self.model.undo()
        self.model.set_selection("gain", 9)
        self.assertFalse(self.model.redo())

    def test_one_drag_is_one_undo_action(self):
        self.model.begin()
        self.model.set_value(0, "gain", 5)
        self.model.set_value(1, "gain", 6)
        self.assertTrue(self.model.commit())
        self.assertEqual(len(self.model.undo_stack), 1)
        self.model.undo()
        self.assertEqual(self.model.frames, self.frames)

    def test_voicing_changes_keep_legal_coefficient_counts(self):
        self.model.set_selection("pitch", 0)
        self.assertEqual(len(self.model.frames[0].k_indices), 4)
        self.model.set_selection("pitch", 160)
        self.assertEqual(len(self.model.frames[0].k_indices), 10)
        self.assertTrue(validate_frame(self.model.frames[0], tables["tms5220"]))
        self.model.set_selection("gain", 0)
        self.assertEqual(self.model.frames[0], QuantizedFrame(0))
        self.model.set_selection("gain", 5)
        self.assertTrue(validate_frame(self.model.frames[0], tables["tms5220"]))

    def test_inactive_coefficients_and_nonfinite_values(self):
        self.model.set_selection("pitch", 0)
        before = self.model.frames
        self.assertFalse(self.model.set_selection("k10", .9))
        self.assertEqual(self.model.frames, before)
        with self.assertRaises(ValueError):
            self.model.set_selection("gain", float("nan"))

    def test_talkie_and_32_entry_pitch_profiles(self):
        for variant in ("Talkie", "tms5100", "tms5110a"):
            model = LpcEditorModel(self.frames, variant)
            model.set_selection("pitch", 1)
            self.assertLess(model.frames[0].pitch_index, len(tables[variant].pitchtable))
            self.assertTrue(validate_frame(model.frames[0], tables[variant]))


if __name__ == "__main__":
    unittest.main()
