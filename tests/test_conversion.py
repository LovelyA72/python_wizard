import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from pywizard.FrameDataBinaryEncoder import BitPacker
from pywizard.GuiConversion import run_conversion_process
from pywizard.TalkieSynthesizer import TalkieSynthesizer
from pywizard.userSettings import settings


class ConversionTests(unittest.TestCase):
    def run_worker(self, filename, optimize=False):
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        process = context.Process(target=run_conversion_process, args=(
            events, str(filename), dict(settings.export_to_odict()),
            dict(enabled=optimize, passes=1, radius=1, lookahead=0,
                 loss_profile="balanced", input_frame_samples=200),
        ))
        process.start()
        try:
            while True:
                event, value = events.get(timeout=30)
                if event != "progress":
                    break
            self.assertEqual(event, "complete", str(value))
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
            return value
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            events.close()
            process.close()

    def test_default_cli_and_spawned_gui_worker_match(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            filename = directory / "speech.wav"
            t = np.arange(1600) / 8000
            wavfile.write(filename, 8000, (.12 * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16))
            output = directory / "speech.lpc"
            subprocess.run([sys.executable, "python_wizard", "-f", "lpc", "-o", str(output), str(filename)],
                           cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, timeout=30)
            processor, chip, *_ = self.run_worker(filename)
            self.assertEqual(chip, "Talkie")
            self.assertEqual(output.read_bytes(), BitPacker.lpc_bytes(processor))
            self.assertEqual(processor.frame_ms, 25)
            self.assertEqual(len(processor.source_pcm), 1600)
            self.assertTrue(TalkieSynthesizer.decode_frames(output.read_bytes())[-1].is_stop)

    def test_qbp_worker_with_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "speech.wav"
            t = np.arange(600) / 8000
            wavfile.write(filename, 8000, (.1 * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16))
            processor, chip, result, _, initial, optimized, _ = self.run_worker(filename, optimize=True)
            self.assertEqual(chip, "Talkie")
            self.assertIsNotNone(result)
            self.assertEqual(len(initial), len(optimized))
            self.assertEqual(len(processor.frames), 4)


if __name__ == "__main__":
    unittest.main()
