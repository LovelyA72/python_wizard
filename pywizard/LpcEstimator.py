"""Frame estimators and sequence-aware LPC analysis plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pywizard.userSettings import settings


LPC_ESTIMATOR_NAMES = ("autocorrelation", "burg", "qbp")


@dataclass(frozen=True)
class LpcAnalysisContext:
    """Original and preemphasized PCM, target tables, and analysis controls.

    Pitch values are periods in source samples. The caller inserts stop frames.
    """

    source: object
    signal: object
    coding_table: object
    pitch_detector: object
    frame_samples: int
    window_width: float
    minimum_pitch_hz: float
    maximum_pitch_hz: float
    override_pitch: float | None = None
    pitch_offset: int = 0

    def pitch_signal(self):
        """Use the existing bandpass, padding inputs shorter than filtfilt."""
        from pywizard.Buffer import Buffer
        from pywizard.Filterer import Filterer

        source = self.source
        if len(source.samples) <= 33:
            source = Buffer(
                samples=np.pad(source.samples, (0, 34 - len(source.samples))),
                sampleRate=source.sampleRate,
            )
        filtered = Filterer(
            source, lowPassCutoffInHZ=self.minimum_pitch_hz,
            highPassCutoffInHZ=self.maximum_pitch_hz, gain=1,
        ).process()
        return Buffer(samples=filtered.samples[:len(self.source.samples)],
                      sampleRate=self.source.sampleRate)


@dataclass
class LpcAnalysis:
    """Frames and selected pitch periods, excluding the stop frame."""

    frames: list
    pitch_periods: np.ndarray


@dataclass(frozen=True)
class LpcEstimate:
    """Reflection coefficients and total residual energy for one audio frame."""

    reflection_coefficients: tuple
    residual_energy: float


class LpcEstimator(ABC):
    """Interface implemented by LPC analysis algorithms."""

    def __init__(self, order=10):
        if order < 1:
            raise ValueError("order must be at least one")
        self.order = order

    @abstractmethod
    def estimate(self, buf):
        """Estimate reflection coefficients and residual energy for ``buf``."""

    def analyze(self, context):
        """Default sequence adapter for existing ``estimate(buf)`` plugins.

        Override this method to own windowing, sequence decisions and
        quantization, returning FrameData or QuantizedFrame objects.
        """
        from pywizard.FrameData import FrameData
        from pywizard.HammingWindow import HammingWindow
        from pywizard.Reflector import Reflector
        from pywizard.Segmenter import Segmenter

        segments = Segmenter(context.signal, context.window_width,
                             frame_samples=context.frame_samples)
        pitches = np.zeros(segments.numberOfSegments())
        if context.override_pitch is not None:
            pitches.fill(context.override_pitch)
        else:
            pitch_segments = Segmenter(context.pitch_signal(), 2,
                                       frame_samples=context.frame_samples)
            for buf, index in pitch_segments.eachSegment():
                pitches[index] = context.pitch_detector.detect(buf)
        frames = []
        for buf, index in segments.eachSegment():
            HammingWindow.processBuffer(buf)
            reflector = Reflector.fromLpcEstimate(
                context.coding_table, self.estimate(buf), buf.size,
            )
            frames.append(FrameData(reflector, pitches[index], repeat=False))
        return LpcAnalysis(frames, pitches)

    def _samples(self, buf):
        samples = np.asarray(buf.samples[buf.start:buf.end], dtype=float)
        if samples.ndim != 1:
            raise ValueError("LPC estimation requires one-dimensional audio")
        if samples.size <= self.order:
            raise ValueError("audio frame must contain more samples than LPC order")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio frame contains non-finite samples")
        return samples

    def _silent_estimate(self):
        return LpcEstimate(
            reflection_coefficients=tuple([0.0] * (self.order + 1)),
            residual_energy=0.0,
        )


class AutocorrelationLpcEstimator(LpcEstimator):
    """LPC analysis using autocorrelation and Leroux-Gueguen recursion."""

    def estimate(self, buf):
        samples = self._samples(buf)
        autocorrelation = np.empty(self.order + 1, dtype=float)
        for lag in range(self.order + 1):
            autocorrelation[lag] = np.dot(
                samples[:samples.size - lag],
                samples[lag:],
            )
        return self.estimate_from_autocorrelation(autocorrelation)

    def estimate_from_autocorrelation(self, autocorrelation):
        autocorrelation = np.asarray(autocorrelation, dtype=float)
        if autocorrelation.size < self.order + 1:
            raise ValueError("not enough autocorrelation coefficients for LPC order")
        if autocorrelation[0] <= np.finfo(float).eps:
            return self._silent_estimate()

        reflection = np.zeros(self.order + 1, dtype=float)
        previous = np.zeros(self.order + 1, dtype=float)
        error = np.zeros(self.order + 2, dtype=float)

        reflection[1] = -autocorrelation[1] / autocorrelation[0]
        error[1] = autocorrelation[1]
        error[2] = autocorrelation[0] + reflection[1] * autocorrelation[1]

        for i in range(2, self.order + 1):
            value = autocorrelation[i]
            previous[1] = value

            for j in range(1, i):
                previous[j + 1] = error[j] + reflection[j] * value
                value += reflection[j] * error[j]
                error[j] = previous[j]

            denominator = error[i]
            if abs(denominator) <= np.finfo(float).eps:
                return LpcEstimate(
                    reflection_coefficients=tuple(reflection),
                    residual_energy=0.0,
                )
            reflection[i] = -value / denominator
            error[i + 1] = denominator + reflection[i] * value
            error[i] = previous[i]

        return LpcEstimate(
            reflection_coefficients=tuple(reflection),
            residual_energy=max(0.0, float(error[self.order + 1])),
        )


class BurgLpcEstimator(LpcEstimator):
    """Native NumPy implementation of Burg LPC analysis."""

    def estimate(self, buf):
        samples = self._samples(buf)
        input_energy = float(np.dot(samples, samples))
        if input_energy <= np.finfo(float).eps:
            return self._silent_estimate()

        reflection = np.zeros(self.order + 1, dtype=float)
        forward_error = samples[1:].copy()
        backward_error = samples[:-1].copy()
        denominator = float(
            np.dot(forward_error, forward_error)
            + np.dot(backward_error, backward_error)
        )
        residual_energy = input_energy
        epsilon = np.finfo(float).eps

        for model_order in range(1, self.order + 1):
            if denominator <= epsilon:
                break

            coefficient = (
                -2.0
                * float(np.dot(backward_error, forward_error))
                / denominator
            )
            coefficient = float(np.clip(coefficient, -1.0 + epsilon, 1.0 - epsilon))
            reflection[model_order] = coefficient
            residual_energy *= max(0.0, 1.0 - coefficient * coefficient)

            previous_forward_error = forward_error
            forward_error = forward_error + coefficient * backward_error
            backward_error = backward_error + coefficient * previous_forward_error

            if model_order == self.order:
                break

            denominator = (
                (1.0 - coefficient * coefficient) * denominator
                - backward_error[-1] * backward_error[-1]
                - forward_error[0] * forward_error[0]
            )
            forward_error = forward_error[1:]
            backward_error = backward_error[:-1]

        return LpcEstimate(
            reflection_coefficients=tuple(reflection),
            residual_energy=max(0.0, float(residual_energy)),
        )


def create_lpc_estimator_from_settings():
    """Create the LPC estimator selected in the application settings."""
    # QBP reuses the solver above, so import it after defining that solver.
    from pywizard.QbpLpcEstimator import QbpLpcEstimator

    estimator_types = {
        "autocorrelation": AutocorrelationLpcEstimator,
        "burg": BurgLpcEstimator,
        "qbp": QbpLpcEstimator,
    }
    try:
        estimator_type = estimator_types[settings.lpcEstimator]
    except KeyError as error:
        raise ValueError(
            "unsupported LPC estimator: {}".format(settings.lpcEstimator)
        ) from error
    return estimator_type()
