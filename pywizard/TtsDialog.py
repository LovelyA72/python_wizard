"""English Flite synthesis and chip preview dialog."""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from pywizard.FliteTts import ENGINES, VOICES, SynthesisCancelled, TtsOptions, TtsSession, check_cancelled


TTS_CONTROLS = (
    ("pitch", "Pitch (semitones)", -12, 12, .5, 1),
    ("formant", "Formants (%)", 70, 130, 1, 100),
    ("speed", "Speed (×)", .5, 2, .05, 1),
    ("intonation", "Intonation (%)", 0, 200, 5, 100),
    ("gain", "Gain (dB)", -24, 12, .5, 1),
)


def play_preview(samples, cancel):
    try:
        import pyaudio
    except ImportError as error:
        raise RuntimeError("PyAudio is required for preview. Install it with:\n"
                           "python -m pip install pyaudio") from error
    audio, stream = None, None
    try:
        check_cancelled(cancel)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=8000, output=True,
                            frames_per_buffer=400)
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        for offset in range(0, len(pcm), 800):
            check_cancelled(cancel)
            stream.write(pcm[offset:offset + 800])
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        finally:
            if audio is not None:
                audio.terminate()


class TtsDialog(tk.Toplevel):
    def __init__(self, parent, engine, text, on_apply, on_close, options=None):
        super().__init__(parent)
        self.title("Text to speech")
        self.transient(parent)
        self.minsize(540, 480)
        self.geometry("640x540")
        self.options = options or TtsOptions()
        self.on_apply, self.on_close = on_apply, on_close
        self.session = TtsSession()
        self.events = queue.SimpleQueue()
        self.cancel = threading.Event()
        self.busy = False
        self.closed = False
        self.worker = None
        self.poll_id = None
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _event: self.close())

        content = ttk.Frame(self, padding=12)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Text").pack(anchor="w")
        editor = ttk.Frame(content)
        editor.pack(fill="both", expand=True, pady=(5, 12))
        self.text = tk.Text(editor, wrap="word", height=8, undo=True)
        scrollbar = ttk.Scrollbar(editor, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.insert("1.0", text)
        tone_controls = ttk.Frame(content)
        tone_controls.pack(fill="x", pady=(0, 8))
        self.process_tone_numbers = tk.BooleanVar(value=self.options.process_tone_numbers)
        self.tone_checkbox = ttk.Checkbutton(
            tone_controls, text="process tone numbers", variable=self.process_tone_numbers)
        self.tone_checkbox.pack(side="left")
        ttk.Label(tone_controls, text="你好 or knee2 how3").pack(side="left", padx=12)
        selectors = ttk.Frame(content)
        selectors.pack(fill="x", pady=(0, 8))
        ttk.Label(selectors, text="Voice").pack(side="left", padx=(0, 6))
        self.voice_labels = {voice["label"]: key for key, voice in VOICES.items()}
        self.voice = tk.StringVar(value=VOICES[self.options.voice]["label"])
        self.voice_selector = ttk.Combobox(selectors, textvariable=self.voice,
                                          values=tuple(self.voice_labels), state="readonly", width=27)
        self.voice_selector.pack(side="left", fill="x", expand=True, padx=(0, 16))
        ttk.Label(selectors, text="Engine").pack(side="left", padx=(0, 6))
        selected = next((label for label, variant in ENGINES.items() if variant == engine), "Talkie")
        self.engine = tk.StringVar(value=selected)
        self.engine_selector = ttk.Combobox(selectors, textvariable=self.engine,
                                            values=tuple(ENGINES), state="readonly", width=12)
        self.engine_selector.pack(side="left")
        adjustments = ttk.Frame(content)
        adjustments.pack(fill="x", pady=(0, 10))
        adjustments.columnconfigure(1, weight=1)
        self.option_vars, self.option_widgets = {}, []
        for row, (name, label, low, high, step, multiplier) in enumerate(TTS_CONTROLS):
            ttk.Label(adjustments, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            variable = tk.DoubleVar(value=getattr(self.options, name) * multiplier)
            self.option_vars[name] = variable
            slider = ttk.Scale(adjustments, from_=low, to=high, variable=variable,
                               command=lambda value, key=name, increment=step: self._move_slider(key, value, increment))
            slider.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=3)
            field = ttk.Spinbox(adjustments, from_=low, to=high, increment=step,
                                textvariable=variable, width=7)
            field.grid(row=row, column=2, sticky="e", pady=3)
            self.option_widgets.extend((slider, field))
        controls = ttk.Frame(content)
        controls.pack(fill="x")
        self.apply_button = ttk.Button(controls, text="Apply", command=self.apply)
        self.apply_button.pack(side="right")
        self.play_button = ttk.Button(controls, text="Play", command=self.play_stop)
        self.play_button.pack(side="right", padx=8)
        self.status = tk.StringVar(value="")
        ttk.Label(content, textvariable=self.status, anchor="w").pack(fill="x", pady=(10, 0))
        self.grab_set()
        self.text.focus_set()

    def _move_slider(self, name, value, step):
        self.option_vars[name].set(round(round(float(value) / step) * step, 2))

    def _read_options(self):
        try:
            values = {name: self.option_vars[name].get() / multiplier
                      for name, _label, _low, _high, _step, multiplier in TTS_CONTROLS}
            options = TtsOptions(voice=self.voice_labels[self.voice.get()],
                                 process_tone_numbers=self.process_tone_numbers.get(), **values)
        except (tk.TclError, KeyError, ValueError) as error:
            raise ValueError("Choose a voice and enter numeric values for the adjustments.") from error
        options.validate()
        return options

    def _set_inputs_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.text.configure(state=state)
        self.tone_checkbox.configure(state=state)
        for selector in (self.voice_selector, self.engine_selector):
            selector.configure(state="readonly" if enabled else "disabled")
        for widget in self.option_widgets:
            widget.configure(state=state)

    def play_stop(self):
        if self.busy:
            self.cancel.set()
            self.play_button.configure(state="disabled")
            self.status.set("Stopping…")
        else:
            self._start("play")

    def apply(self):
        if not self.busy:
            self._start("apply")

    def _start(self, action):
        text = self.text.get("1.0", "end-1c")
        engine = ENGINES[self.engine.get()]
        if not text.strip():
            self.status.set("Enter text to synthesize.")
            self.text.focus_set()
            return
        try:
            self.options = self._read_options()
        except ValueError as error:
            self.status.set(str(error))
            return
        self.busy = True
        self.cancel = threading.Event()
        self._set_inputs_enabled(False)
        self.apply_button.configure(state="disabled")
        self.play_button.configure(text="Stop", state="normal")
        self.status.set("Synthesizing…")
        self.worker = threading.Thread(target=self._run, args=(
            text, engine, action, self.cancel, self.options), daemon=True)
        self.worker.start()
        self.poll_id = self.after(50, self._poll)

    def _run(self, text, engine, action, cancel, options):
        try:
            result = self.session.synthesize(text, engine, cancel, options=options)
            check_cancelled(cancel)
            if action == "apply":
                self.events.put(("apply", result))
                return
            self.events.put(("playing", None))
            play_preview(result.audio, cancel)
            self.events.put(("finished", None))
        except SynthesisCancelled:
            self.events.put(("finished", None))
        except Exception as error:
            self.events.put(("error", str(error)))

    def _idle(self):
        self.busy = False
        self._set_inputs_enabled(True)
        self.apply_button.configure(state="normal")
        self.play_button.configure(text="Play", state="normal")

    def _poll(self):
        self.poll_id = None
        if self.closed:
            return
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "playing":
                if not self.cancel.is_set():
                    self.status.set("Playing…")
                continue
            self._idle()
            if event == "apply" and not self.cancel.is_set():
                try:
                    self.on_apply(value)
                except Exception as error:
                    self.status.set("Apply failed")
                    messagebox.showerror("Apply failed", str(error), parent=self)
                else:
                    self.close()
                    return
            elif event == "error":
                self.status.set("Synthesis or playback failed")
                messagebox.showerror("Text to speech", value, parent=self)
            else:
                self.status.set("Stopped" if self.cancel.is_set() else "")
        if self.busy:
            self.poll_id = self.after(50, self._poll)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        if self.poll_id is not None:
            self.after_cancel(self.poll_id)
        text = self.text.get("1.0", "end-1c")
        try:
            self.options = self._read_options()
        except ValueError:
            pass
        self.grab_release()
        self.destroy()
        self.on_close(text)
