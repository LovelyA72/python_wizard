import unittest
from unittest.mock import patch

import numpy as np

from pywizard.Buffer import Buffer
from pywizard.PitchDetector import (
    AutocorrelationPitchDetector,
    PitchDetector,
    YinPitchDetector,
    create_pitch_detector_from_settings,
)
from pywizard.PitchEstimator import PitchEstimator
from pywizard.Processor import Processor
from pywizard.userSettings import settings


class _FixedPitchDetector(PitchDetector):
    def __init__(self, period):
        self.period = period
        self.buffers = []

    def detect(self, buf):
        self.buffers.append(buf)
        return self.period


class _Segmenter:
    def __init__(self, buf, windowWidth):
        self.buf = buf

    def numberOfSegments(self):
        return 2

    def eachSegment(self):
        yield Buffer(sampleRate=8000, samples=[0.0] * 64), 0
        yield Buffer(sampleRate=8000, samples=[0.0] * 64), 1


class PitchDetectorTests(unittest.TestCase):
    def test_processor_uses_injected_detector_for_each_pitch_frame(self):
        detector = _FixedPitchDetector(42.5)
        processor = Processor.__new__(Processor)
        processor.pitchDetector = detector
        source = Buffer(sampleRate=8000, samples=[0.0] * 128)

        with patch("pywizard.Processor.Filterer") as filterer, patch(
            "pywizard.Processor.Segmenter", _Segmenter
        ):
            filterer.return_value.process.return_value = source
            table = processor.pitchTableForBuffer(source)

        self.assertEqual(table.tolist(), [42.5, 42.5])
        self.assertEqual(len(detector.buffers), 2)

    def test_autocorrelation_detector_passes_its_configuration(self):
        detector = AutocorrelationPitchDetector(70, 350, 0.75)
        buf = Buffer(sampleRate=8000, samples=[0.0] * 64)

        with patch.object(PitchEstimator, "pitchForPeriod", return_value=23.0) as estimate:
            result = detector.detect(buf)

        self.assertEqual(result, 23.0)
        estimate.assert_called_once_with(
            buf,
            minimum_pitch_hz=70,
            maximum_pitch_hz=350,
            sub_multiple_threshold=0.75,
        )

    def test_pitch_estimator_can_be_configured_without_global_settings(self):
        estimator = PitchEstimator.__new__(PitchEstimator)
        estimator.buf = Buffer(sampleRate=8000, samples=[0.0] * 64)
        estimator.minimum_pitch_hz = 80
        estimator.maximum_pitch_hz = 400

        self.assertEqual(estimator.minimumPeriod(), 19)
        self.assertEqual(estimator.maximumPeriod(), 101)

    def test_yin_detects_a_sine_wave_period(self):
        sample_rate = 8000
        frequency = 223.0
        samples = np.sin(
            2 * np.pi * frequency * np.arange(400, dtype=float) / sample_rate
        )
        buf = Buffer(sampleRate=sample_rate, samples=samples)

        period = YinPitchDetector(50, 500).detect(buf)

        self.assertAlmostEqual(period, sample_rate / frequency, delta=0.15)

    def test_yin_detects_a_harmonic_signal_fundamental(self):
        sample_rate = 8000
        frequency = 125.0
        time = np.arange(480, dtype=float) / sample_rate
        samples = (
            0.3 * np.sin(2 * np.pi * frequency * time)
            + np.sin(2 * np.pi * 2 * frequency * time)
        )
        buf = Buffer(sampleRate=sample_rate, samples=samples)

        period = YinPitchDetector(50, 500).detect(buf)

        self.assertAlmostEqual(period, sample_rate / frequency, delta=0.25)

    def test_yin_returns_zero_for_silence(self):
        buf = Buffer(sampleRate=8000, samples=np.zeros(400))

        self.assertEqual(YinPitchDetector(50, 500).detect(buf), 0.0)

    def test_yin_returns_zero_when_noise_has_no_acceptable_trough(self):
        samples = np.random.default_rng(1234).normal(0, 0.1, 400)
        buf = Buffer(sampleRate=8000, samples=samples)

        self.assertEqual(YinPitchDetector(50, 500).detect(buf), 0.0)

    def test_yin_returns_zero_when_frame_is_too_short_for_pitch_range(self):
        samples = np.sin(2 * np.pi * 200 * np.arange(10) / 8000)
        buf = Buffer(sampleRate=8000, samples=samples)

        self.assertEqual(YinPitchDetector(50, 500).detect(buf), 0.0)

    def test_factory_creates_selected_yin_detector(self):
        with patch.object(settings, "pitchDetector", "yin"), patch.object(
            settings, "yinThreshold", 0.2
        ):
            detector = create_pitch_detector_from_settings()

        self.assertIsInstance(detector, YinPitchDetector)
        self.assertEqual(detector.threshold, 0.2)

    def test_yin_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            YinPitchDetector(50, 500, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
