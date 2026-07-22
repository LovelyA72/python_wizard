import unittest
from unittest.mock import patch

import numpy as np

from pywizard.Buffer import Buffer
from pywizard.LpcEstimator import (
    AutocorrelationLpcEstimator,
    BurgLpcEstimator,
    LpcEstimate,
    LpcEstimator,
    create_lpc_estimator_from_settings,
)
from pywizard.Processor import Processor
from pywizard.Reflector import Reflector
from pywizard.userSettings import settings


class _CodingTable:
    kStopFrameIndex = 15
    rms = tuple(range(16))


class _FixedLpcEstimator(LpcEstimator):
    def __init__(self):
        super().__init__()
        self.buffers = []

    def estimate(self, buf):
        self.buffers.append(buf)
        return LpcEstimate(
            reflection_coefficients=tuple([0.0] * 11),
            residual_energy=1.0,
        )


class LpcEstimatorTests(unittest.TestCase):
    def test_autocorrelation_estimator_preserves_legacy_results(self):
        rng = np.random.default_rng(2468)
        samples = rng.normal(0, 0.15, 200)
        samples += 0.4 * np.sin(2 * np.pi * 0.07 * np.arange(200))
        buf = Buffer(sampleRate=8000, samples=samples)

        estimate = AutocorrelationLpcEstimator().estimate(buf)
        reflector = Reflector.fromLpcEstimate(_CodingTable(), estimate, buf.size)

        expected_reflection = [
            0.0,
            -0.7259799181064557,
            0.06178675696292009,
            0.2994937051635553,
            0.3797689528320801,
            0.3228423738977194,
            0.32689613599575584,
            0.1781018654557277,
            0.1781505709505676,
            0.08299097458094538,
            0.06390114859965033,
        ]
        np.testing.assert_allclose(reflector.ks, expected_reflection, rtol=1e-12)
        self.assertAlmostEqual(reflector._rms, 5779.083219280205)

    def test_legacy_reflector_entry_point_uses_autocorrelation_estimator(self):
        samples = np.sin(2 * np.pi * 200 * np.arange(200) / 8000)
        buf = Buffer(sampleRate=8000, samples=samples)

        expected = Reflector.fromLpcEstimate(
            _CodingTable(),
            AutocorrelationLpcEstimator().estimate(buf),
            buf.size,
        )
        actual = Reflector.translateCoefficients(
            _CodingTable(),
            buf.getCoefficientsFor(),
            buf.size,
        )

        np.testing.assert_allclose(actual.ks, expected.ks)
        self.assertAlmostEqual(actual._rms, expected._rms)

    def test_burg_recovers_second_order_autoregressive_model(self):
        rng = np.random.default_rng(9876)
        noise = rng.normal(0, 1, 5000)
        samples = np.zeros_like(noise)
        for index in range(2, samples.size):
            samples[index] = (
                1.5 * samples[index - 1]
                - 0.7 * samples[index - 2]
                + noise[index]
            )
        buf = Buffer(sampleRate=8000, samples=samples[500:])

        estimate = BurgLpcEstimator(order=2).estimate(buf)

        self.assertAlmostEqual(estimate.reflection_coefficients[1], -1.5 / 1.7, delta=0.03)
        self.assertAlmostEqual(estimate.reflection_coefficients[2], 0.7, delta=0.03)
        self.assertLess(estimate.residual_energy, buf.energy())

    def test_burg_returns_silent_estimate_for_silence(self):
        buf = Buffer(sampleRate=8000, samples=np.zeros(200))

        estimate = BurgLpcEstimator().estimate(buf)

        self.assertEqual(estimate.reflection_coefficients, tuple([0.0] * 11))
        self.assertEqual(estimate.residual_energy, 0.0)

    def test_burg_reflection_coefficients_are_stable(self):
        samples = np.random.default_rng(123).normal(0, 1, 200)
        estimate = BurgLpcEstimator().estimate(
            Buffer(sampleRate=8000, samples=samples)
        )

        self.assertTrue(
            np.all(np.abs(estimate.reflection_coefficients[1:]) < 1)
        )

    def test_factory_creates_selected_burg_estimator(self):
        with patch.object(settings, "lpcEstimator", "burg"):
            estimator = create_lpc_estimator_from_settings()

        self.assertIsInstance(estimator, BurgLpcEstimator)

    def test_processor_uses_injected_lpc_estimator_for_each_frame(self):
        estimator = _FixedLpcEstimator()
        buf = Buffer(sampleRate=8000, samples=np.ones(200) * 0.1)

        with patch.object(settings, "preEmphasis", False), patch.object(
            settings, "includeExplicitStopFrame", False
        ), patch.object(settings, "overridePitch", True), patch.object(
            settings, "pitchValue", 40
        ):
            processor = Processor(
                buf,
                "tms5100",
                lpc_estimator=estimator,
            )

        self.assertEqual(len(estimator.buffers), 1)
        self.assertEqual(processor.frames[0].reflector.ks, [0.0] * 11)

    def test_lpc_estimator_requires_a_valid_order(self):
        with self.assertRaises(ValueError):
            BurgLpcEstimator(order=0)


if __name__ == "__main__":
    unittest.main()
