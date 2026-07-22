import unittest

import numpy as np

from lpcplayer.tables import tables, tms5220
from pywizard.CodingTable import CodingTable
from pywizard.FrameDataBinaryEncoder import BitPacker
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TalkieSynthesizer import TalkieSynthesizer
from pywizard.TmsSynthesizer import TmsSynthesizer


class _Processor:
    def __init__(self, frames, variant="Talkie"):
        self.frames = frames
        self.codingTable = CodingTable(variant)


class TalkieSynthesizerTests(unittest.TestCase):
    def test_talkie_conversion_is_exact_tms5220_alias(self):
        self.assertIs(tables["Talkie"], tms5220)
        self.assertIs(tables["talkie"], tms5220)
        self.assertEqual(CodingTable("Talkie").bits, CodingTable("tms5220").bits)

    def test_stream_round_trip_preserves_tms5220_frames(self):
        frames = [
            QuantizedFrame(7, False, 0, (10, 12, 6, 7)),
            QuantizedFrame(9, False, 20, (18, 15, 8, 9, 7, 6, 5, 4, 3, 2)),
            QuantizedFrame(8, True, 21),
            QuantizedFrame(0),
            QuantizedFrame(15),
        ]
        encoded = BitPacker.lpc_bytes(_Processor(frames))
        tms_encoded = BitPacker.lpc_bytes(_Processor(frames, "tms5220"))
        self.assertEqual(encoded, tms_encoded)
        self.assertEqual(TalkieSynthesizer.decode_frames(encoded), frames)

    def test_preview_is_talkie_not_existing_tms5220_synthesis(self):
        frames = [
            QuantizedFrame(10, False, 25, (20, 15, 8, 8, 8, 8, 8, 4, 4, 4)),
            QuantizedFrame(10, True, 25),
        ]
        talkie = TalkieSynthesizer().synthesize(frames)
        chip = TmsSynthesizer(tms5220).synthesize(frames)
        self.assertEqual(talkie.shape, (400,))
        self.assertTrue(np.all(np.isfinite(talkie)))
        self.assertFalse(np.array_equal(talkie, chip))

    def test_packed_preview_matches_indexed_preview(self):
        frames = [
            QuantizedFrame(6, False, 0, (12, 13, 7, 8)),
            QuantizedFrame(15),
        ]
        synth = TalkieSynthesizer()
        encoded = BitPacker.lpc_bytes(_Processor(frames))
        expected = synth.synthesize_u8(frames)
        np.testing.assert_array_equal(synth.decode(encoded), expected)


if __name__ == "__main__":
    unittest.main()
