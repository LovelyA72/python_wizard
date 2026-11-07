"""QBOX-informed sequence analysis, not a bit-exact QBOX Pro port."""

import numpy as np

from pywizard.Buffer import Buffer
from pywizard.LpcEstimator import AutocorrelationLpcEstimator, LpcAnalysis
from pywizard.QuantizedFrame import QuantizedFrame


class QbpLpcEstimator(AutocorrelationLpcEstimator):
    """Separate voiced/unvoiced windows, pitch repair, and quantized gain.

    ``estimate`` remains the ordinary autocorrelation frame solver. The
    sequence override supplies the additional analysis stages. Thresholds and
    smoothing rules are new choices based on the recovered architecture.
    """

    def __init__(self, order=10):
        if order != 10:
            raise ValueError("qbp requires ten LPC coefficients")
        super().__init__(order)

    @staticmethod
    def _window(samples, center, length):
        start = center - length // 2
        stop = start + length
        result = np.zeros(length, dtype=float)
        first, last = max(0, start), min(len(samples), stop)
        if first < last:
            result[first - start:last - start] = samples[first:last]
        return result

    @staticmethod
    def _smooth_pitch(periods, rms, silence_floor):
        """Repair isolated voicing gaps and octave jumps on a copied track."""
        result = np.asarray(periods, dtype=float).copy()
        result[~np.isfinite(result) | (result < 0) | (rms <= silence_floor)] = 0
        original = result.copy()
        for index in range(1, len(result) - 1):
            left, current, right = original[index - 1:index + 2]
            if rms[index] <= silence_floor:
                continue
            if left > 0 and right > 0 and max(left, right) / min(left, right) < 1.25:
                reference = (left + right) / 2
                if current == 0:
                    result[index] = reference
                else:
                    choices = (current / 2, current, current * 2)
                    result[index] = min(choices, key=lambda p: abs(np.log(p / reference)))
            elif left == 0 and right == 0:
                result[index] = 0
        return result

    def analyze(self, context):
        from lpcplayer.tables import tables

        source = np.asarray(context.source.samples, dtype=float)
        signal = np.asarray(context.signal.samples, dtype=float)
        hop = context.frame_samples
        count = (len(source) + hop - 1) // hop
        centers = np.arange(count) * hop + hop // 2
        rms = np.array([
            np.sqrt(np.mean(source[i * hop:min((i + 1) * hop, len(source))] ** 2))
            for i in range(count)
        ])
        silence_floor = max(1e-6, float(np.max(rms)) * 0.001)
        if context.override_pitch is not None:
            if not np.isfinite(context.override_pitch) or context.override_pitch < 0:
                raise ValueError("pitch override must be a nonnegative finite period")
            periods = np.full(count, context.override_pitch)
        else:
            filtered = context.pitch_signal().samples
            pitch_length = max(
                self.order + 2, 2 * hop,
                int(np.ceil(2 * context.source.sampleRate / context.minimum_pitch_hz)) + 2,
            )
            periods = np.zeros(count)
            for index, center in enumerate(centers):
                if rms[index] <= silence_floor:
                    continue
                window = self._window(filtered, center, pitch_length)
                window -= np.mean(window)
                threshold = 0.3 * np.max(np.abs(window))
                clipped = np.sign(window) * np.maximum(np.abs(window) - threshold, 0)
                periods[index] = context.pitch_detector.detect(Buffer(
                    samples=clipped, sampleRate=context.source.sampleRate,
                ))
            periods = self._smooth_pitch(periods, rms, silence_floor)
            minimum = context.source.sampleRate / context.maximum_pitch_hz
            maximum = context.source.sampleRate / context.minimum_pitch_hz
            periods[(periods < minimum) | (periods > maximum)] = 0

        chip = tables[context.coding_table.name]
        if context.coding_table.name.lower() == "talkie":
            from pywizard.TalkieSynthesizer import ENERGY, PERIOD, K_TABLES, CHIRP
            energy_table, pitch_table = ENERGY, PERIOD
            k_tables = [
                np.asarray(values, dtype=float) / (32768 if i < 2 else 128)
                for i, values in enumerate(K_TABLES)
            ]
            chirp = np.asarray(CHIRP, dtype=float) / 256
        else:
            energy_table, pitch_table = chip.energytable, chip.pitchtable
            k_tables = [np.asarray(values, dtype=float) / 512 for values in chip.ktable]
            chirp = np.asarray(chip.chirptable, dtype=float)
            chirp = np.where(chirp > 127, chirp - 256, chirp) / 64

        frames = []
        for index, center in enumerate(centers):
            if rms[index] <= silence_floor:
                frames.append(QuantizedFrame(energy_index=0))
                periods[index] = 0
                continue
            period = periods[index]
            pitch_index = 0
            if period > 0:
                pitch_index = 1 + int(np.argmin(np.abs(np.asarray(pitch_table[1:]) - period)))
                if context.override_pitch is None:
                    pitch_index = int(np.clip(pitch_index + context.pitch_offset, 1, len(pitch_table) - 1))
            voiced = pitch_index != 0
            # Default windowWidth=2 gives 150% voiced and 200% unvoiced.
            width = context.window_width * (0.75 if voiced else 1.0)
            length = max(self.order + 2, int(round(hop * width)))
            samples = self._window(signal if voiced else source, center, length)
            estimate = self.estimate(Buffer(
                samples=samples * np.hamming(length),
                sampleRate=context.source.sampleRate,
            ))
            active = 10 if voiced else 4
            indices = tuple(
                int(np.argmin(np.abs(k_tables[k] - estimate.reflection_coefficients[k + 1])))
                for k in range(active)
            )
            quantized_k = np.array([k_tables[k][v] for k, v in enumerate(indices)])
            # Approximate residual RMS using the actual quantized filter and
            # target excitation.
            residual_rms = rms[index] * np.sqrt(np.prod(np.maximum(1e-9, 1 - quantized_k ** 2)))
            excitation_rms = 1.0
            if voiced:
                selected_period = int(pitch_table[pitch_index])
                excitation_rms = np.sqrt(np.sum(chirp[:selected_period] ** 2) / selected_period)
            gain = 512 * residual_rms / max(excitation_rms, 1e-9)
            # Speech excludes both silence index zero and stop index fifteen.
            energy_index = 1 + int(np.argmin(np.abs(np.asarray(energy_table[1:15]) - gain)))
            frames.append(QuantizedFrame(energy_index, False, pitch_index, indices))
        return LpcAnalysis(frames, periods)
