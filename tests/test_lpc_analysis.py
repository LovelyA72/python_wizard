import hashlib
import unittest

import numpy as np

from pywizard.Buffer import Buffer
from pywizard.FrameDataBinaryEncoder import BitPacker
from pywizard.FrameOptimizer import validate_frame
from pywizard.LpcEstimator import AutocorrelationLpcEstimator, LpcAnalysis
from pywizard.Processor import Processor
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.QbpLpcEstimator import QbpLpcEstimator
from pywizard.TalkieSynthesizer import TalkieSynthesizer
from pywizard.userSettings import settings
from lpcplayer.tables import tables


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.saved = settings.__dict__.copy()
        settings.frameRate = 25
        settings.windowWidth = 2
        settings.minimumPitchInHZ = 50
        settings.maximumPitchInHZ = 500
        settings.unvoicedThreshold = 0.7
        settings.overridePitch = False
        settings.pitchOffset = 0
        settings.includeExplicitStopFrame = True
        settings.preEmphasisAlpha = -0.9373
        settings.subMultipleThreshold = 0.9
        settings.yinThreshold = 0.1
        settings.pitchDetector = "yin"
        settings.preEmphasis = True
        settings.lpcEstimator = "qbp"
        t = np.arange(2400) / 8000
        self.samples = .18 * np.sin(2 * np.pi * 130 * t) + .04 * np.sin(2 * np.pi * 390 * t)

    def tearDown(self):
        settings.__dict__.clear()
        settings.__dict__.update(self.saved)

    def process(self, samples=None, model="tms5220", estimator=None):
        return Processor(Buffer(samples=self.samples if samples is None else samples,
                                sampleRate=8000), model, lpc_estimator=estimator)

    def test_legacy_streams_match_pre_refactor_baselines(self):
        # Captured from the unmodified pipeline, using the same synthetic input.
        expected = (
            "da109d77fde013ea0e546321213f418c4c0605925be7db6e38e3e6d060e823ce",
            "77831e402fd5779bed9d697a41f43f64b56298d8638e33eaad2ee87385a9bcd5",
            "4c1b3cfead165e17dd6285c248875302f5f48e0cb23d7bbb30629de6931970c6",
            "1315dfba85c2762449a81b3f7b104501396dc7675266b5f4ee9f496593eb5759",
            "47ee192606daa363b6e76ab6d593b036b6fa155a320b302300c570c4b8ce18c1",
            "f9fddcf4ab2683ece3079cb32928095d5f9f5f2619a1382962f668a29f4a9ba0",
            "886f90f6cc72bbfd738beb63e7da3ac97e4c22e0dc01945ba113c658b7bd0917",
            "4df371ce40a2a3457d92a0a806410f8da39a1a70a1168e523884ab1101d4006c",
        )
        actual = []
        for estimator in ("autocorrelation", "burg"):
            for detector in ("autocorrelation", "yin"):
                for emphasis in (False, True):
                    settings.lpcEstimator, settings.pitchDetector = estimator, detector
                    settings.preEmphasis = emphasis
                    actual.append(hashlib.sha256(BitPacker.lpc_bytes(self.process())).hexdigest())
        self.assertEqual(tuple(actual), expected)

    def test_frame_only_plugin_remains_supported(self):
        class RecordingEstimator(AutocorrelationLpcEstimator):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def estimate(self, buf):
                self.calls += 1
                return super().estimate(buf)

        plugin = RecordingEstimator()
        processor = self.process(estimator=plugin)
        self.assertEqual(plugin.calls, 12)
        self.assertTrue(QuantizedFrame.from_frame(processor.frames[-1]).is_stop)

    def test_sequence_plugin_receives_unwindowed_owned_pcm(self):
        source = self.samples.copy()

        class SequenceEstimator(AutocorrelationLpcEstimator):
            def analyze(plugin, context):
                np.testing.assert_array_equal(context.source.samples, source)
                self.assertFalse(np.shares_memory(context.source.samples, self.samples))
                return LpcAnalysis([QuantizedFrame(0)], np.zeros(1))

        processor = self.process(estimator=SequenceEstimator())
        self.assertEqual(processor.frames, [QuantizedFrame(0), QuantizedFrame(15)])
        np.testing.assert_array_equal(self.samples, source)

    def test_qbp_streams_valid_for_each_target(self):
        for model in ("tms5100", "tms5110a", "tms5200", "tms5220", "Talkie"):
            with self.subTest(model=model):
                processor = self.process(model=model)
                self.assertEqual(len(processor.frames), 13)
                self.assertTrue(all(validate_frame(f, tables[model]) for f in processor.frames))
                self.assertEqual(processor.frames[-1], QuantizedFrame(15))
                self.assertEqual(BitPacker.lpc_bytes(processor), BitPacker.lpc_bytes(self.process(model=model)))
                if model in ("tms5220", "Talkie"):
                    self.assertEqual(TalkieSynthesizer.decode_frames(BitPacker.lpc_bytes(processor)), processor.frames)

    def test_silence_short_input_and_no_stop(self):
        for length in (1, 10, 201):
            processor = self.process(samples=np.zeros(length))
            self.assertTrue(all(f.is_silence for f in processor.frames[:-1]))
        settings.includeExplicitStopFrame = False
        self.assertFalse(any(f.is_stop for f in self.process().frames))

    def test_pitch_override_zero_means_unvoiced(self):
        settings.overridePitch = True
        for period, voiced in ((0, False), (62, True)):
            settings.pitchValue = period
            frames = self.process().frames[:-1]
            self.assertTrue(all(f.is_voiced == voiced for f in frames))

    def test_smoothing_repairs_gap_and_octave_without_voicing_silence(self):
        rms = np.ones(7)
        rms[5] = 0
        track = QbpLpcEstimator._smooth_pitch([60, 0, 62, 124, 62, 0, 62], rms, .001)
        self.assertEqual(track[1], 61)
        self.assertEqual(track[3], 62)
        self.assertEqual(track[5], 0)

    def test_invalid_pcm_rejected(self):
        for samples in (np.array([]), np.array([np.nan]), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                self.process(samples=samples)


if __name__ == "__main__":
    unittest.main()
