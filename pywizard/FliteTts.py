"""Flite acoustic parameters to chip frames, without intermediate speech PCM."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np

from pywizard.CodingTable import CodingTable
from pywizard.GuiConversion import ConvertedProcessor
from pywizard.LpcEstimator import AutocorrelationLpcEstimator
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.TmsSynthesizer import TmsSynthesizer, SynthesizerState
from pywizard.TalkieSynthesizer import TalkieSynthesizer, TalkieState


ENGINES = {
    "TMS5100": "tms5100",
    "TMS5110A": "tms5110a",
    "TMS5200": "tms5200",
    "TMS5220": "tms5220",
    "Talkie": "Talkie",
}
VOICES = {voice["id"]: voice for voice in json.loads(
    (Path(__file__).parent / "flite_voices.json").read_text(encoding="utf-8"))}
DEFAULT_VOICE = next(iter(VOICES))
FLITE_EXECUTABLE = "python-wizard-flite" + (".exe" if os.name == "nt" else "")
MISSING_FLITE_MESSAGE = f"Can not find bin/{FLITE_EXECUTABLE}! Did you run setup.py?"


@dataclass(frozen=True)
class TtsOptions:
    voice: str = DEFAULT_VOICE
    pitch: float = 0.0
    formant: float = 1.0
    speed: float = 1.0
    intonation: float = 1.0
    gain: float = 0.0
    process_tone_numbers: bool = False

    def validate(self):
        if not isinstance(self.process_tone_numbers, bool):
            raise ValueError("Process tone numbers must be on or off.")
        if self.voice not in VOICES:
            raise ValueError("Choose an available voice.")
        for name, low, high in (
            ("pitch", -12, 12), ("formant", .7, 1.3), ("speed", .5, 2),
            ("intonation", 0, 2), ("gain", -24, 12),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name.capitalize()} must be between {low:g} and {high:g}.")


class SynthesisCancelled(Exception):
    pass


def check_cancelled(cancel):
    if cancel.is_set():
        raise SynthesisCancelled()


@dataclass(frozen=True)
class AcousticParameters:
    sample_rate: int
    alpha: float
    frame_shift: float
    frames: np.ndarray
    tone_marks: np.ndarray | None = None

    @classmethod
    def from_json(cls, output):
        try:
            data = json.loads(output)
            frames = np.asarray(data["frames"], dtype=float)
            rate, alpha, shift = int(data["sample_rate"]), float(data["alpha"]), float(data["frame_shift"])
            if data.get("version") != 2:
                raise ValueError("Rebuild the Flite helper with: python -m pywizard.build_flite")
            if (rate < 8000 or not -1 < alpha < 1
                    or not 0 < shift <= .1 or frames.ndim != 2
                    or frames.shape[0] < 1 or frames.shape[1] < 3
                    or not np.all(np.isfinite(frames)) or np.any(frames[:, 0] < 0)):
                raise ValueError("invalid acoustic parameters")
            marks = data.get("tone_marks")
            if marks is not None:
                marks = np.asarray(marks, dtype=float)
                if (marks.shape != (len(frames), 2) or not np.all(np.isfinite(marks))
                        or np.any(marks != np.floor(marks)) or np.any(marks < 0)
                        or np.any(marks[:, 1] > 5)
                        or np.any((marks[:, 0] == 0) != (marks[:, 1] == 0))):
                    raise ValueError("invalid tone alignment")
                marks = marks.astype(np.int64)
                for group in np.unique(marks[:, 0]):
                    if group and len(np.unique(marks[marks[:, 0] == group, 1])) != 1:
                        raise ValueError("inconsistent tone alignment")
            return cls(rate, alpha, shift, frames, marks)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Flite parameters: {error}") from error


class FliteBackend:
    def __init__(self, executable=None):
        self.executable = executable

    def resolve_executable(self):
        name = FLITE_EXECUTABLE
        bundled = Path(__file__).parent / "bin" / name
        executable = self.executable or os.environ.get("PYWIZARD_FLITE_HELPER")
        executable = executable or (str(bundled) if bundled.is_file() else shutil.which(name))
        if not executable or not (Path(executable).is_file() or shutil.which(str(executable))):
            raise RuntimeError(MISSING_FLITE_MESSAGE)
        return executable

    def parameters(self, text, cancel, voice=DEFAULT_VOICE, speed=1.0, process_tone_numbers=False):
        TtsOptions(voice=voice, speed=speed).validate()
        executable = self.resolve_executable()
        check_cancelled(cancel)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        command = [str(executable), "--voice", voice, "--speed", str(speed)]
        if process_tone_numbers:
            command.append("--process-tone-numbers")
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, **options)
        except FileNotFoundError as error:
            raise RuntimeError(MISSING_FLITE_MESSAGE) from error
        with process:
            started = time.monotonic()
            pending = text.encode("utf-8")
            try:
                while True:
                    check_cancelled(cancel)
                    if time.monotonic() - started > 120:
                        raise RuntimeError("Flite synthesis timed out. Try a shorter passage.")
                    try:
                        output, errors = process.communicate(pending, timeout=.1)
                        break
                    except subprocess.TimeoutExpired:
                        pending = None
            except BaseException:
                process.kill()
                process.communicate()
                raise
            if process.returncode:
                reason = errors.decode("utf-8", errors="replace").strip()
                raise RuntimeError("Flite synthesis failed" + (": " + reason if reason else "."))
        check_cancelled(cancel)
        parameters = AcousticParameters.from_json(output)
        if process_tone_numbers and parameters.tone_marks is None:
            raise ValueError("Rebuild the Flite helper with: python setup.py")
        return parameters


@dataclass
class TtsResult:
    text: str
    engine: str
    processor: ConvertedProcessor
    audio: np.ndarray
    options: TtsOptions = TtsOptions()


TONE_CONTOURS = {
    1: ((0, 1), (4, 4)),
    2: ((0, 1), (-2, 4)),
    3: ((0, .5, 1), (-2, -5, 1)),
    4: ((0, 1), (5, -4)),
    5: ((0, 1), (0, -1)),
}


def _apply_tone_contours(pitches, marks, reference_pitch, options):
    """Apply literal tone targets over each marked token's voiced span."""
    for group in np.unique(marks[:, 0]):
        if group == 0:
            continue
        indices = np.flatnonzero((marks[:, 0] == group) & (pitches > 0))
        if not len(indices):
            continue
        tone = int(marks[indices[0], 1])
        positions, semitones = TONE_CONTOURS[tone]
        progress = (indices - indices[0]) / max(1, indices[-1] - indices[0])
        contour = np.interp(progress, positions, semitones)
        pitches[indices] = reference_pitch * 2 ** (
            (options.pitch + options.intonation * contour) / 12)


def _spectral_frames(parameters, cancel, options=TtsOptions()):
    """Project the warped cepstral envelope onto the target's 0–4 kHz band."""
    frames = parameters.frames
    duration = len(frames) * parameters.frame_shift
    count = max(1, int(np.ceil(round(duration / .025, 6))))
    source_times = (np.arange(len(frames)) + .5) * parameters.frame_shift
    target_times = (np.arange(count) + .5) * .025
    interpolated = np.column_stack([
        np.interp(target_times, source_times, frames[:, k])
        for k in range(1, frames.shape[1])
    ])
    # Voicing is categorical. Interpolating across a boundary invents low F0.
    nearest = np.minimum((target_times / parameters.frame_shift).astype(int), len(frames) - 1)
    pitches = frames[nearest, 0].copy()
    voiced = pitches > 0
    if np.any(voiced) and (options.pitch != 0 or options.intonation != 1):
        log_pitch = np.log(pitches[voiced])
        center = np.median(np.log(frames[frames[:, 0] > 0, 0]))
        pitches[voiced] = np.exp(center + options.intonation * (log_pitch - center)
                                 + options.pitch * np.log(2) / 12)
    if options.process_tone_numbers:
        if parameters.tone_marks is None:
            raise ValueError("Rebuild the Flite helper with: python setup.py")
        if np.any(voiced):
            reference_pitch = float(np.median(frames[frames[:, 0] > 0, 0]))
            _apply_tone_contours(pitches, parameters.tone_marks[nearest], reference_pitch, options)
    frequency = np.linspace(0, 4000, 257)
    source_frequency = frequency / options.formant
    z = np.exp(-2j * np.pi * np.minimum(source_frequency, parameters.sample_rate / 2)
               / parameters.sample_rate)
    warped = (z - parameters.alpha) / (1 - parameters.alpha * z)
    basis = np.real(warped[None, :] ** np.arange(interpolated.shape[1])[:, None])
    # The chip's chirp has DC content. Preemphasis prevents the fitted vocal
    # tract from spending most of its output headroom amplifying that offset.
    preemphasis = abs(1 - .9373 * np.exp(-2j * np.pi * frequency / 8000)) ** 2
    correlations = []
    for coefficients in interpolated:
        check_cancelled(cancel)
        log_magnitude = coefficients @ basis
        power = np.exp(np.clip(2 * log_magnitude, -60, 60)) / 32768 ** 2
        power *= preemphasis
        power[source_frequency > parameters.sample_rate / 2] = 0
        correlations.append(np.fft.irfft(power, n=512)[:11])
    correlations = np.asarray(correlations)
    # One utterance-wide gain preserves relative phoneme levels.
    level = np.sqrt(np.maximum(correlations[:, 0], 0))
    scale = .25 / max(float(np.percentile(level, 95)), 1e-9)
    return correlations * scale ** 2, pitches


def quantize_parameters(parameters, engine, cancel, options=TtsOptions()):
    from lpcplayer.tables import tables
    from pywizard.TalkieSynthesizer import ENERGY, PERIOD, K_TABLES, CHIRP
    from pywizard.TalkieMitigator import silence_transition_frame

    if engine not in ENGINES.values():
        raise ValueError("Choose an available chip or engine.")
    options.validate()
    talkie = engine == "Talkie"
    chip = tables[engine]
    if talkie:
        energy_table, pitch_table = np.asarray(ENERGY), np.asarray(PERIOD)
        k_tables = [np.asarray(t) / (32768 if i < 2 else 128) for i, t in enumerate(K_TABLES)]
        chirp = np.asarray(CHIRP) / 256
    else:
        energy_table, pitch_table = np.asarray(chip.energytable), np.asarray(chip.pitchtable)
        k_tables = [np.asarray(t) / 512 for t in chip.ktable]
        chirp = np.asarray(chip.chirptable)
        chirp = np.where(chirp > 127, chirp - 256, chirp) / 64
    correlations, pitches = _spectral_frames(parameters, cancel, options)
    gain_multiplier = 10 ** (options.gain / 20)
    estimator = AutocorrelationLpcEstimator()
    frames, gains, gain_limits = [], [], []
    for correlation, f0 in zip(correlations, pitches):
        check_cancelled(cancel)
        rms = float(np.sqrt(max(0, correlation[0])))
        if rms < .002:
            frame = QuantizedFrame(0)
            gain = 0.0
            gain_limit = 0.0
        else:
            pitch = 0 if f0 <= 0 else 1 + int(np.argmin(abs(pitch_table[1:] - 8000 / f0)))
            estimate = estimator.estimate_from_autocorrelation(correlation)
            active = 10 if pitch else 4
            ks = tuple(int(np.argmin(abs(k_tables[k] - estimate.reflection_coefficients[k + 1])))
                       for k in range(active))
            reflection = np.array([k_tables[k][v] for k, v in enumerate(ks)])
            period = int(pitch_table[pitch]) + int(talkie) if pitch else 0
            response, peak_response = _excitation_response(reflection, chirp, period)
            gain_limit = .75 * 512 / max(peak_response, 1e-9)
            gain = min(gain_multiplier * 512 * rms / max(response, 1e-9), gain_limit)
            energy = 1 + int(np.argmin(abs(energy_table[1:15] - gain)))
            frame = QuantizedFrame(energy, False, pitch, ks)
        frames.append(frame)
        gains.append(gain)
        gain_limits.append(gain_limit)

    # Fit gain to the filter's steady response instead of chasing the previous
    # frame's interpolated level. Smooth before quantizing the energy track.
    for index in range(1, len(frames) - 1):
        frame = frames[index]
        neighbors = frames[index - 1:index + 2]
        if frame.is_silence or any(f.is_silence or f.is_voiced != frame.is_voiced for f in neighbors):
            continue
        gain = min(gain_limits[index], np.exp(np.dot(
            (.25, .5, .25), np.log(np.maximum(gains[index - 1:index + 2], 1e-9)))))
        energy = 1 + int(np.argmin(abs(energy_table[1:15] - gain)))
        frames[index] = replace(frame, energy_index=energy)

    synth = TalkieSynthesizer() if talkie else TmsSynthesizer(chip)
    state = TalkieState() if talkie else SynthesizerState()
    output, chunks = [], []
    for frame in frames:
        check_cancelled(cancel)
        audio, next_state = synth.synthesize_frame(frame, state)
        if frame.is_silence:
            if np.max(abs(audio)) > .02:
                if talkie:
                    frame = silence_transition_frame()
                else:
                    neutral = tuple(int(np.argmin(abs(t))) for t in k_tables)
                    frame = QuantizedFrame(1, False, 1, neutral)
                audio, next_state = synth.synthesize_frame(frame, state)
        output.append(frame)
        chunks.append(audio)
        state = next_state
    output.append(QuantizedFrame(15))
    return ConvertedProcessor(output, CodingTable(engine), None, 25.0), np.concatenate(chunks)


def _excitation_response(reflection, chirp, period):
    """Unit excitation's AC level and peak for gain and headroom calibration."""
    if not period:
        response = 1 / np.sqrt(np.prod(1 - reflection ** 2))
        return response, 4 * response
    denominator = np.array([1.0])
    for coefficient in reflection:
        denominator = np.r_[denominator, 0] + coefficient * np.r_[0, denominator[::-1]]
    excitation = np.zeros(period)
    length = min(period, len(chirp))
    excitation[:length] = chirp[:length]
    z = np.exp(-2j * np.pi * np.arange(period) / period)
    response = sum(coefficient * z ** order for order, coefficient in enumerate(denominator))
    cycle = np.fft.ifft(np.fft.fft(excitation) / response).real
    return float(np.std(cycle)), float(np.max(abs(cycle)))


class TtsSession:
    """Keep acoustic inference cached across preview and engine changes."""

    def __init__(self, backend=None):
        self.backend = backend or FliteBackend()
        self.text = None
        self.parameters = None
        self.results = {}
        self.acoustic_options = (DEFAULT_VOICE, 1.0, False)

    def synthesize(self, text, engine, cancel=None, options=None):
        options = options or TtsOptions()
        options.validate()
        cancel = cancel or threading.Event()
        text = text.strip()
        if not text or not any(character.isalpha() or character.isdigit() for character in text):
            raise ValueError("Enter text to synthesize.")
        if len(text) > 10000:
            raise ValueError("Use at most 10,000 characters per synthesis.")
        if engine not in ENGINES.values():
            raise ValueError("Choose an available chip or engine.")
        check_cancelled(cancel)
        acoustic_options = (options.voice, options.speed, options.process_tone_numbers)
        if text != self.text or acoustic_options != self.acoustic_options:
            backend_options = dict(voice=options.voice, speed=options.speed)
            spoken_text = text
            if options.process_tone_numbers:
                from pywizard.MandarinText import mandarin_to_english
                spoken_text = mandarin_to_english(text)
                check_cancelled(cancel)
                backend_options["process_tone_numbers"] = True
            parameters = self.backend.parameters(spoken_text, cancel, **backend_options)
            self.text, self.parameters, self.results = text, parameters, {}
            self.acoustic_options = acoustic_options
        if engine not in self.results or self.results[engine].options != options:
            processor, audio = quantize_parameters(self.parameters, engine, cancel, options)
            check_cancelled(cancel)
            self.results[engine] = TtsResult(text, engine, processor, audio, options)
        return self.results[engine]
