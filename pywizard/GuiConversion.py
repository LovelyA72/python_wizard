"""Process-isolated conversion helpers used by the Tk GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lpcplayer.tables import tables
from pywizard.Buffer import Buffer
from pywizard.CodingTable import CodingTable
from pywizard.FrameOptimizer import OptimizationConfig, OptimizationResult, optimize_frames
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


def optimize_processor(
    processor: Processor,
    original_pcm: np.ndarray,
    chip_variant: str,
    passes: int,
    radius: int,
    lookahead: int,
    loss_profile: str,
    input_frame_samples: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[OptimizationResult, list[QuantizedFrame], np.ndarray, np.ndarray]:
    """Run the shared optimizer and retain its before/after comparison audio."""
    initial_frames = [QuantizedFrame.from_frame(frame) for frame in processor.frames]
    config = OptimizationConfig(
        passes=passes, radius=radius, lookahead=lookahead,
        loss_profile=loss_profile, input_frame_samples=input_frame_samples,
    )
    chip_tables = tables[chip_variant]
    result = optimize_frames(
        initial_frames, original_pcm, chip_tables, config,
        progress_callback=progress_callback,
    )
    processor.frames = result.frames
    return (
        result,
        initial_frames,
        synthesize_frames(initial_frames, chip_variant),
        synthesize_frames(result.frames, chip_variant),
    )


def run_conversion_process(
    event_queue,
    filename: str,
    settings_values: dict[str, object],
    optimizer_values: dict[str, object],
) -> None:
    """Analyze and optionally optimize in a separate Python process."""
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
        result = initial_frames = initial_audio = optimized_audio = None
        talkie_mitigation: TalkieMitigationResult | None = None

        if chip_variant.lower() == "talkie":
            talkie_mitigation = mitigate_talkie_frames(
                [QuantizedFrame.from_frame(frame) for frame in analyzed.frames],
                original_pcm,
                TalkieMitigationConfig(
                    strength=float(settings.talkieMitigationStrength),
                    gate_threshold=float(settings.talkieGateThreshold),
                    input_frame_samples=max(1, int(round(8 * settings.frameRate))),
                ),
            )
            analyzed.frames = talkie_mitigation.frames

        if optimizer_values["enabled"]:
            def report_progress(current: int, total: int) -> None:
                event_queue.put(("progress", (current, total)))

            result, initial_frames, initial_audio, optimized_audio = optimize_processor(
                analyzed,
                original_pcm,
                chip_variant,
                int(optimizer_values["passes"]),
                int(optimizer_values["radius"]),
                int(optimizer_values["lookahead"]),
                str(optimizer_values["loss_profile"]),
                int(optimizer_values["input_frame_samples"]),
                report_progress,
            )
            if chip_variant.lower() == "talkie":
                talkie_mitigation = mitigate_talkie_frames(
                    result.frames,
                    original_pcm,
                    TalkieMitigationConfig(
                        strength=float(settings.talkieMitigationStrength),
                        gate_threshold=float(settings.talkieGateThreshold),
                        input_frame_samples=max(
                            1, int(round(8 * settings.frameRate)),
                        ),
                    ),
                )
                analyzed.frames = talkie_mitigation.frames
                result.frames = analyzed.frames
                optimized_audio = TalkieSynthesizer().synthesize(
                    analyzed.frames,
                )

        processor = ConvertedProcessor(analyzed.frames, analyzed.codingTable)
        event_queue.put(("complete", (
            processor, chip_variant, result, initial_frames,
            initial_audio, optimized_audio, talkie_mitigation,
        )))
    except Exception as error:
        event_queue.put(("error", str(error)))
