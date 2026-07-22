import unittest

import numpy as np

from lpcplayer.player import LpcDecoder
from lpcplayer.tables import tms5220
from pywizard.FrameDataBinaryEncoder import BitPacker
from pywizard.FrameOptimizer import (
    OptimizationConfig,
    neighbor_indices,
    optimize_frames,
    validate_frame,
)
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TmsSynthesizer import TmsSynthesizer


class _Processor:
    codingTable = type(
        "Table", (), {
            "bits": (4, 1, 6, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3),
            "kStopFrameIndex": 15,
            "parameters": staticmethod(lambda: (
                "kParameterGain", "kParameterRepeat", "kParameterPitch",
                "kParameterK1", "kParameterK2", "kParameterK3", "kParameterK4",
                "kParameterK5", "kParameterK6", "kParameterK7", "kParameterK8",
                "kParameterK9", "kParameterK10",
            )),
        },
    )()

    def __init__(self, frames):
        self.frames = frames


def voiced(energy=8, pitch=20, delta=0):
    sizes = [len(table) for table in tms5220.ktable]
    return QuantizedFrame(
        energy, False, pitch,
        tuple(min(size - 1, size // 2 + delta) for size in sizes),
    )


class OptimizerTests(unittest.TestCase):
    def test_synthesizer_is_deterministic(self):
        frames = [voiced(), QuantizedFrame(5, False, 0, (10, 12, 6, 7))]
        synth = TmsSynthesizer(tms5220)
        np.testing.assert_array_equal(synth.synthesize(frames), synth.synthesize(frames))

    def test_neighbor_indices_clamp_and_deduplicate(self):
        self.assertEqual(neighbor_indices(0, 4, 2), (0, 1, 2))
        self.assertEqual(neighbor_indices(3, 4, 2), (1, 2, 3))

    def test_validation_rejects_bad_indices(self):
        self.assertFalse(validate_frame(QuantizedFrame(16), tms5220))
        self.assertFalse(validate_frame(QuantizedFrame(4, False, 0, (32, 0, 0, 0)), tms5220))
        self.assertTrue(validate_frame(QuantizedFrame(0), tms5220))
        self.assertTrue(validate_frame(QuantizedFrame(15), tms5220))

    def test_silence_and_stop_are_preserved(self):
        frames = [QuantizedFrame(0), voiced(), QuantizedFrame(15)]
        target = TmsSynthesizer(tms5220).synthesize(frames)
        result = optimize_frames(frames, target, tms5220, OptimizationConfig(passes=1))
        self.assertEqual(result.frames[0], frames[0])
        self.assertEqual(result.frames[-1], frames[-1])
        self.assertTrue(all(validate_frame(frame, tms5220) for frame in result.frames))

    def test_indexed_frames_keep_existing_bitstream_bytes(self):
        frames = [voiced(), QuantizedFrame(4, False, 0, (10, 12, 6, 7)), QuantizedFrame(15)]
        before = BitPacker.lpc_bytes(_Processor(frames))
        frozen = [QuantizedFrame.from_frame(frame) for frame in frames]
        after = BitPacker.lpc_bytes(_Processor(frozen))
        self.assertEqual(before, after)

    def test_optimized_stream_decodes(self):
        frames = [voiced(), voiced(energy=7, pitch=22), QuantizedFrame(15)]
        target = TmsSynthesizer(tms5220).synthesize([voiced(energy=9), frames[1]])
        result = optimize_frames(frames, target, tms5220, OptimizationConfig(passes=1))
        decoded = LpcDecoder(tms5220).decode(BitPacker.lpc_bytes(_Processor(result.frames)))
        self.assertGreater(len(decoded), 0)

    def test_optimization_loss_never_increases_and_is_reproducible(self):
        frames = [voiced(energy=7, pitch=19), voiced(energy=6, pitch=21)]
        target_frames = [voiced(energy=8, pitch=20), voiced(energy=7, pitch=22)]
        target = TmsSynthesizer(tms5220).synthesize(target_frames)
        config = OptimizationConfig(passes=2, radius=1)
        first = optimize_frames(frames, target, tms5220, config)
        second = optimize_frames(frames, target, tms5220, config)
        self.assertLessEqual(first.optimized_loss, first.initial_loss + 1e-12)
        self.assertTrue(all(after < before for before, after in first.accepted_loss_pairs))
        self.assertEqual(first.frames, second.frames)
        self.assertEqual(first.initial_loss, second.initial_loss)
        self.assertEqual(first.optimized_loss, second.optimized_loss)

    def test_optimizer_reports_monotonic_progress(self):
        frames = [voiced(), voiced(energy=7)]
        target = TmsSynthesizer(tms5220).synthesize(frames)
        updates = []
        optimize_frames(
            frames, target, tms5220, OptimizationConfig(passes=1),
            progress_callback=lambda current, total: updates.append((current, total)),
        )
        self.assertEqual(updates[0][0], 0)
        self.assertEqual(updates[-1][0], updates[-1][1])
        self.assertEqual(updates, sorted(updates))


if __name__ == "__main__":
    unittest.main()
