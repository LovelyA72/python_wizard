import runpy
import unittest
from pathlib import Path
import multiprocessing
import tempfile

import numpy as np
from scipy.io import wavfile

from lpcplayer.tables import tms5220
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TmsSynthesizer import TmsSynthesizer
from pywizard.GuiConversion import run_conversion_process
from pywizard.userSettings import settings


class GuiOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gui_path = Path(__file__).resolve().parents[1] / "python_wizard_gui"
        cls.gui_module = runpy.run_path(str(gui_path), run_name="python_wizard_gui_test")

    def test_finite_choice_settings_have_dropdown_values(self):
        choices = self.gui_module["SELECTOR_CHOICES"]
        self.assertEqual(choices["pitchDetector"], ("autocorrelation", "yin"))
        self.assertEqual(choices["lpcEstimator"], ("autocorrelation", "burg"))
        self.assertEqual(choices["outputFormat"], ("arduino", "C", "hex", "python"))
        self.assertIn("tms5220", choices["tablesVariant"])
        self.assertIn("Talkie", choices["tablesVariant"])

    def test_gui_exposes_optimizer_and_export_actions(self):
        gui = self.gui_module["Gui"]
        self.assertTrue(hasattr(gui, "_make_optimizer_settings"))
        self.assertTrue(hasattr(gui, "_file_export_optimization_report"))
        self.assertTrue(hasattr(gui, "_file_export_optimization_parameters"))

    def test_ascii_progress_bar_is_bounded(self):
        gui = self.gui_module["Gui"]
        self.assertEqual(gui._progress_bar(1, 4, "Working", width=4), "[#---]  25% Working")
        self.assertEqual(gui._progress_bar(9, 4, "Done", width=4), "[####] 100% Done")

    def test_gui_optimizer_uses_shared_indexed_frame_pipeline(self):
        class Processor:
            frames = [QuantizedFrame(6, False, 0, (10, 12, 6, 7))]

        processor = Processor()
        target = TmsSynthesizer(tms5220).synthesize(processor.frames)
        result, initial, initial_audio, optimized_audio = self.gui_module[
            "optimize_processor_for_gui"
        ](processor, target, "tms5220", 0, 1, 1, "balanced", 200)
        self.assertEqual(processor.frames, result.frames)
        self.assertEqual(initial, result.frames)
        np.testing.assert_array_equal(initial_audio, optimized_audio)

    def test_conversion_worker_crosses_spawn_process_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            wave_path = str(Path(directory) / "short.wav")
            samples = np.asarray(
                12000 * np.sin(2 * np.pi * 120 * np.arange(400) / 8000),
                dtype=np.int16,
            )
            wavfile.write(wave_path, 8000, samples)
            values = dict(settings.export_to_odict())
            values.update({
                "preEmphasis": False,
                "includeExplicitStopFrame": False,
                "tablesVariant": "tms5220",
                "outputFormat": "python",
            })
            for enabled in (False, True):
                with self.subTest(optimization_enabled=enabled):
                    optimizer = {
                        "enabled": enabled,
                        "passes": 1,
                        "radius": 1,
                        "lookahead": 1,
                        "loss_profile": "balanced",
                        "input_frame_samples": 200,
                    }
                    context = multiprocessing.get_context("spawn")
                    events = context.Queue()
                    process = context.Process(
                        target=run_conversion_process,
                        args=(events, wave_path, values, optimizer),
                    )
                    process.start()
                    received = []
                    while True:
                        event = events.get(timeout=30)
                        received.append(event[0])
                        if event[0] in ("complete", "error"):
                            break
                    process.join(timeout=30)
                    events.close()
                    self.assertEqual(process.exitcode, 0)
                    self.assertNotIn("error", received)
                    self.assertEqual("progress" in received, enabled)
                    self.assertEqual(received[-1], "complete")


if __name__ == "__main__":
    unittest.main()
