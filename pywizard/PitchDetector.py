"""Pluggable pitch detector implementations.

Pitch detectors operate on a single audio frame and return the fundamental
period in samples.  Keeping this small interface independent of ``Processor``
makes it possible to add other algorithms, such as YIN, without changing the
LPC processing pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np

from pywizard.PitchEstimator import PitchEstimator
from pywizard.userSettings import settings


class PitchDetector(ABC):
    """Interface implemented by pitch detection algorithms."""

    @abstractmethod
    def detect(self, buf):
        """Return the fundamental period in samples, or ``0.0`` if absent."""


class AutocorrelationPitchDetector(PitchDetector):
    """The autocorrelation-based detector historically used by pywizard."""

    def __init__(self, minimum_pitch_hz, maximum_pitch_hz, sub_multiple_threshold):
        self.minimum_pitch_hz = minimum_pitch_hz
        self.maximum_pitch_hz = maximum_pitch_hz
        self.sub_multiple_threshold = sub_multiple_threshold

    @classmethod
    def from_settings(cls):
        """Build a detector from the current application settings."""
        return cls(
            minimum_pitch_hz=settings.minimumPitchInHZ,
            maximum_pitch_hz=settings.maximumPitchInHZ,
            sub_multiple_threshold=settings.subMultipleThreshold,
        )

    def detect(self, buf):
        return PitchEstimator.pitchForPeriod(
            buf,
            minimum_pitch_hz=self.minimum_pitch_hz,
            maximum_pitch_hz=self.maximum_pitch_hz,
            sub_multiple_threshold=self.sub_multiple_threshold,
        )


class YinPitchDetector(PitchDetector):
    """Fundamental-period detector implementing the YIN algorithm."""

    def __init__(self, minimum_pitch_hz, maximum_pitch_hz, threshold=0.1):
        if minimum_pitch_hz <= 0:
            raise ValueError("minimum_pitch_hz must be greater than zero")
        if maximum_pitch_hz <= minimum_pitch_hz:
            raise ValueError("maximum_pitch_hz must exceed minimum_pitch_hz")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")

        self.minimum_pitch_hz = minimum_pitch_hz
        self.maximum_pitch_hz = maximum_pitch_hz
        self.threshold = threshold

    @classmethod
    def from_settings(cls):
        """Build a detector from the current application settings."""
        return cls(
            minimum_pitch_hz=settings.minimumPitchInHZ,
            maximum_pitch_hz=settings.maximumPitchInHZ,
            threshold=settings.yinThreshold,
        )

    def detect(self, buf):
        samples = np.asarray(buf.samples, dtype=float)
        if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
            return 0.0

        samples = samples - np.mean(samples)
        silence_floor = np.finfo(float).eps * samples.size
        if np.dot(samples, samples) <= silence_floor:
            return 0.0

        minimum_period = max(2, int(np.ceil(buf.sampleRate / self.maximum_pitch_hz)))
        maximum_period = min(
            int(np.floor(buf.sampleRate / self.minimum_pitch_hz)),
            samples.size // 2,
        )
        if maximum_period < minimum_period:
            return 0.0

        difference = self._difference_function(samples, maximum_period)
        normalized = self._cumulative_mean_normalized_difference(difference)
        period = self._absolute_threshold(
            normalized,
            minimum_period,
            maximum_period,
        )
        if period is None:
            return 0.0

        return self._parabolic_interpolation(normalized, period)

    @staticmethod
    def _difference_function(samples, maximum_period):
        """Calculate squared differences using equal overlap for every lag."""
        window_size = samples.size - maximum_period
        reference = samples[:window_size]
        difference = np.zeros(maximum_period + 1, dtype=float)

        for period in range(1, maximum_period + 1):
            delta = reference - samples[period:period + window_size]
            difference[period] = np.dot(delta, delta)

        return difference

    @staticmethod
    def _cumulative_mean_normalized_difference(difference):
        normalized = np.ones_like(difference)
        cumulative_sum = np.cumsum(difference[1:])
        periods = np.arange(1, difference.size, dtype=float)
        normalized[1:] = np.divide(
            difference[1:] * periods,
            cumulative_sum,
            out=np.ones_like(cumulative_sum),
            where=cumulative_sum > np.finfo(float).eps,
        )
        return normalized

    def _absolute_threshold(self, normalized, minimum_period, maximum_period):
        """Find the first acceptable trough, preferring its local minimum."""
        period = minimum_period
        while period <= maximum_period:
            if normalized[period] < self.threshold:
                while (
                    period < maximum_period
                    and normalized[period + 1] < normalized[period]
                ):
                    period += 1
                return period
            period += 1
        return None

    @staticmethod
    def _parabolic_interpolation(normalized, period):
        if period <= 0 or period >= normalized.size - 1:
            return float(period)

        left = normalized[period - 1]
        middle = normalized[period]
        right = normalized[period + 1]
        denominator = left - 2 * middle + right
        if abs(denominator) <= np.finfo(float).eps:
            return float(period)

        adjustment = 0.5 * (left - right) / denominator
        return float(period + np.clip(adjustment, -1.0, 1.0))


def create_pitch_detector_from_settings():
    """Create the pitch detector selected in the application settings."""
    detector_types = {
        "autocorrelation": AutocorrelationPitchDetector,
        "yin": YinPitchDetector,
    }
    try:
        detector_type = detector_types[settings.pitchDetector]
    except KeyError as error:
        raise ValueError(
            "unsupported pitch detector: {}".format(settings.pitchDetector)
        ) from error
    return detector_type.from_settings()
