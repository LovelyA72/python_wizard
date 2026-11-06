import unittest

import numpy as np

from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TalkieMitigator import TalkieMitigationConfig, mitigate_talkie_frames
from pywizard.TalkieSynthesizer import TalkieSynthesizer
from pywizard.FrameOptimizer import validate_frame
from lpcplayer.tables import tables


class TalkieQbpTests(unittest.TestCase):
    def test_voiced_guard_limits_resonant_bursts_without_changing_timing(self):
        frames = [
            QuantizedFrame(14, False, 5, (24, 8, 2, 9, 11, 10, 3, 3, 4, 6)),
            QuantizedFrame(13, False, 8, (8, 15, 10, 11, 6, 8, 8, 5, 5, 6)),
            QuantizedFrame(10, False, 14, (1, 24, 12, 12, 6, 8, 6, 3, 4, 5)),
            QuantizedFrame(9, False, 15, (1, 23, 11, 11, 9, 9, 6, 4, 3, 3)),
            QuantizedFrame(15),
        ]
        source = .06 * np.sin(2 * np.pi * 220 * np.arange(800) / 8000)
        legacy = mitigate_talkie_frames(
            frames, source, TalkieMitigationConfig(strength=2, gate_threshold=0),
        )
        self.assertEqual(legacy.frames, frames)
        protected = mitigate_talkie_frames(
            frames, source,
            TalkieMitigationConfig(strength=2, gate_threshold=0, protect_voiced=True),
        )
        before = TalkieSynthesizer().synthesize(frames)
        after = TalkieSynthesizer().synthesize(protected.frames)
        self.assertEqual(len(before), len(after))
        self.assertLess(np.sqrt(np.mean(after ** 2)), np.sqrt(np.mean(before ** 2)) * .5)
        self.assertTrue(all(a.pitch_index == b.pitch_index for a, b in zip(frames, protected.frames)))
        self.assertTrue(all(validate_frame(f, tables["Talkie"]) for f in protected.frames))
        self.assertTrue(any(c["kind"] == "stabilize" for c in protected.changes))

    def test_disabling_guard_preserves_frames(self):
        frames = [QuantizedFrame(14, False, 20, (0,) * 10), QuantizedFrame(15)]
        result = mitigate_talkie_frames(
            frames, np.ones(200) * .05,
            TalkieMitigationConfig(strength=0, gate_threshold=0, protect_voiced=True),
        )
        self.assertEqual(result.frames, frames)


if __name__ == "__main__":
    unittest.main()
