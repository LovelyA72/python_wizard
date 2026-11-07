"""Process-isolated conversion helpers used by the Tk GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from lpcplayer.tables import tables
from pywizard.Buffer import Buffer
from pywizard.CodingTable import CodingTable
from pywizard.Processor import Processor
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TmsSynthesizer import TmsSynthesizer
from pywizard.TalkieSynthesizer import TalkieSynthesizer
from pywizard.TalkieMitigator import (
    TalkieMitigationConfig,
    TalkieMitigationResult,
    mitigate_talkie_frames,
)
from pywizard.userSettings import settings


@dataclass
class ConvertedProcessor:
    """Minimal picklable processor interface required by the GUI and packer."""

    frames: list[QuantizedFrame] | list[object]
    codingTable: CodingTable
    source_pcm: np.ndarray | None = None
    frame_ms: float = 25.0


def synthesize_frames(
    frames: list[QuantizedFrame] | list[object],
    chip_variant: str,
) -> np.ndarray:
    """Synthesize the currently processed frames for GUI WAV export."""
    quantized = [
        frame
        if isinstance(frame, QuantizedFrame)
        else QuantizedFrame.from_frame(frame)
        for frame in frames
    ]
    synthesizer = (
        TalkieSynthesizer()
        if chip_variant.lower() == "talkie"
        else TmsSynthesizer(tables[chip_variant])
    )
    return synthesizer.synthesize(quantized)


def run_conversion_process(
    event_queue,
    filename: str,
    settings_values: dict[str, object],
) -> None:
    """Analyze a WAV in a separate Python process."""
    try:
        errors = settings.import_from_dict(settings_values)
        if errors:
            raise ValueError("Invalid settings: {}".format(", ".join(errors)))
        chip_variant = str(settings_values["tablesVariant"])
        buffer = Buffer.fromWave(filename)
        if buffer is None:
            raise ValueError("The WAV must be mono, 8-bit or 16-bit audio.")
        original_pcm = buffer.samples.copy()
        analyzed = Processor(buffer, chip_variant)
        talkie_mitigation: TalkieMitigationResult | None = None

        if chip_variant.lower() == "talkie":
            talkie_mitigation = mitigate_talkie_frames(
                [QuantizedFrame.from_frame(frame) for frame in analyzed.frames],
                original_pcm,
                TalkieMitigationConfig(
                    strength=float(settings.talkieMitigationStrength),
                    gate_threshold=float(settings.talkieGateThreshold),
                    input_frame_samples=max(1, int(round(8 * settings.frameRate))),
                    protect_voiced=settings.lpcEstimator == "qbp",
                ),
            )
            analyzed.frames = talkie_mitigation.frames

        processor = ConvertedProcessor(
            analyzed.frames, analyzed.codingTable, original_pcm, float(settings.frameRate),
        )
        event_queue.put(("complete", (
            processor, chip_variant, talkie_mitigation,
        )))
    except Exception as error:
        event_queue.put(("error", str(error)))


def write_synthesized_wav(path: str | Path, audio: np.ndarray) -> None:
    """Write normalized 8 kHz, signed 16-bit audio."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak:
        audio = audio / max(1.0, peak)
    wavfile.write(str(path), 8000, np.asarray(np.clip(audio, -1, 1) * 32767, dtype=np.int16))
