"""Tk LPC track editor. Edits commit to the current conversion per gesture."""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from pywizard.LpcEditorModel import LpcEditorModel


class LpcEditor(tk.Toplevel):
    LEFT = 64
    RIGHT = 14
    TOP = 26
    BOTTOM = 22

    def __init__(self, parent, frames, chip_variant, source, frame_ms,
                 on_change, on_play, on_close):
        model = LpcEditorModel(frames, chip_variant)
        super().__init__(parent)
        self.transient(parent)
        self.title("LPC editor")
        self.geometry("1080x690")
        self.minsize(740, 540)
        self.model = model
        self.source = np.asarray(source, dtype=float)
        self.source_frame_ms = frame_ms
        self.frame_ms = 25.0  # Preview/output runs at 8 kHz, 200 samples/frame.
        self.on_change, self.on_play, self.on_close = on_change, on_play, on_close
        self.first, self.visible = 0, self.model.count
        self.cursor = 0
        self.mode = tk.StringVar(value="Select")
        self.coefficient = tk.StringVar(value="K1")
        self.track = tk.StringVar(value="gain")
        self.value = tk.StringVar(value="0")
        self.readout = tk.StringVar()
        self._drag = None
        self._drawing = False
        self._last_point = None

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill="x")
        for mode in ("Select", "Draw"):
            ttk.Radiobutton(toolbar, text=mode, value=mode, variable=self.mode).pack(side="left", padx=(0, 8))
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo, width=6)
        self.undo_button.pack(side="left", padx=(10, 2))
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo, width=6)
        self.redo_button.pack(side="left", padx=2)
        for text, command in (("+", lambda: self.zoom(0.5)), ("−", lambda: self.zoom(2)), ("All", self.show_all)):
            ttk.Button(toolbar, text=text, command=command, width=4).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Play selection", command=self.play_selection).pack(side="right")
        ttk.Button(toolbar, text="Play all", command=lambda: self.on_play(self.model.frames, None)).pack(side="right", padx=6)

        self.canvases = {}
        for name, height in (("Wave", 85), ("Gain", 100), ("Pitch · Hz", 110), ("K", 145)):
            row = ttk.Frame(self)
            row.pack(fill="both", expand=True, padx=8)
            if name == "K":
                selector = ttk.Combobox(row, textvariable=self.coefficient,
                                        values=tuple(f"K{i}" for i in range(1, 11)),
                                        state="readonly", width=4)
                selector.place(x=0, y=5)
                selector.bind("<<ComboboxSelected>>", self._coefficient_changed)
            canvas = tk.Canvas(row, background="white", highlightthickness=0, height=height)
            canvas.pack(fill="both", expand=True)
            if name == "K":
                selector.lift()
            self.canvases[name] = canvas
            canvas.bind("<Configure>", lambda _e: self.redraw())
            canvas.bind("<ButtonPress-1>", lambda e, n=name: self._press(e, n))
            canvas.bind("<B1-Motion>", lambda e, n=name: self._motion(e, n))
            canvas.bind("<ButtonRelease-1>", self._release)
        self.scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.scroll)
        self.scrollbar.pack(fill="x", padx=(72, 22), pady=4)

        bottom = ttk.Frame(self, padding=(8, 5))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.readout).pack(side="left")
        ttk.Button(bottom, text="Unvoiced", command=lambda: self.set_selection("pitch", 0)).pack(side="right", padx=3)
        ttk.Button(bottom, text="Silence", command=lambda: self.set_selection("gain", 0)).pack(side="right", padx=3)
        ttk.Button(bottom, text="Set", command=self.set_numeric, width=5).pack(side="right", padx=3)
        self.value_entry = ttk.Entry(bottom, textvariable=self.value, width=9)
        self.value_entry.pack(side="right", padx=3)
        self.value_entry.bind("<Return>", lambda _e: self.set_numeric())
        ttk.Label(bottom, textvariable=self.track, width=5).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-y>", lambda _e: self.redo())
        self.bind("<Left>", lambda e: self.move_cursor(-1, e))
        self.bind("<Right>", lambda e: self.move_cursor(1, e))
        self.bind("<space>", self._space)
        self.after_idle(self.redraw)
        self._sync_value()

    def _coefficient_changed(self, _event):
        self.track.set(self.coefficient.get().lower())
        self._sync_value()
        self.redraw()

    def close(self):
        self._release(None)
        self.on_close()
        self.destroy()

    def _space(self, event):
        if not isinstance(event.widget, (ttk.Entry, ttk.Combobox)):
            self.play_selection()
            return "break"

    def _track_name(self, name):
        return self.coefficient.get().lower() if name == "K" else ("pitch" if name.startswith("Pitch") else "gain")

    def _frame_at(self, canvas, x):
        width = max(1, canvas.winfo_width() - self.LEFT - self.RIGHT)
        return max(self.first, min(self.first + self.visible - 1,
                   self.first + int((x - self.LEFT) * self.visible / width)))

    def _scale(self, name):
        if name == "Gain":
            return 0, 14
        if name.startswith("Pitch"):
            maximum = max(8000 / p for p in self.model.pitch_table if p)
            return 0, max(600, int(np.ceil(maximum / 100)) * 100)
        return -1, 1

    def _value_at(self, canvas, name, y):
        low, high = self._scale(name)
        height = max(1, canvas.winfo_height() - self.TOP - self.BOTTOM)
        fraction = np.clip((y - self.TOP) / height, 0, 1)
        return float(high - fraction * (high - low))

    def _press(self, event, name):
        self.focus_set()
        canvas = self.canvases[name]
        frame = self._frame_at(canvas, event.x)
        self.cursor = frame
        self._drawing = self.mode.get() == "Draw" and name != "Wave" and not (event.state & 1)
        self._drag = frame
        self._last_point = None
        if name != "Wave":
            self.track.set(self._track_name(name))
        if self._drawing:
            self.model.begin()
            self.model.select(frame, frame)
            self._draw_value(canvas, name, event.y, frame)
        else:
            self.model.select(frame, frame)
            self._sync_value()
        self.redraw()

    def _draw_value(self, canvas, name, y, frame):
        value = self._value_at(canvas, name, y)
        if self._last_point is not None:
            previous, old_value = self._last_point
            for index in range(min(previous, frame), max(previous, frame) + 1):
                fraction = 1 if frame == previous else (index - previous) / (frame - previous)
                self.model.set_value(index, self._track_name(name), old_value + fraction * (value - old_value))
        else:
            self.model.set_value(frame, self._track_name(name), value)
        self._last_point = (frame, value)
        self._sync_value()

    def _motion(self, event, name):
        if self._drag is None:
            return
        canvas = self.canvases[name]
        self.cursor = self._frame_at(canvas, event.x)
        if self._drawing:
            self._draw_value(canvas, name, event.y, self.cursor)
        else:
            self.model.select(self._drag, self.cursor)
            self._sync_value()
        self.redraw()

    def _release(self, _event):
        changed = self.model.commit()
        self._drag = None
        self._drawing = False
        self._last_point = None
        if changed:
            self.on_change(self.model.frames)
        self.redraw()

    def _sync_value(self):
        value = self.model.values(self.track.get())[self.cursor]
        self.value.set("" if value is None else f"{value:.4g}")

    def set_numeric(self):
        try:
            self.set_selection(self.track.get(), float(self.value.get()))
        except ValueError as error:
            messagebox.showerror("Invalid value", str(error), parent=self)

    def set_selection(self, track, value):
        if self.model.set_selection(track, value):
            self.on_change(self.model.frames)
        self._sync_value()
        self.redraw()

    def undo(self):
        if self.model.undo():
            self.on_change(self.model.frames)
        self._sync_value()
        self.redraw()

    def redo(self):
        if self.model.redo():
            self.on_change(self.model.frames)
        self._sync_value()
        self.redraw()

    def play_selection(self):
        self.on_play(self.model.frames, self.model.selection)

    def move_cursor(self, delta, event):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox)):
            return
        self.cursor = max(0, min(self.model.count - 1, self.cursor + delta))
        self.model.select(self.cursor, self.cursor)
        if not self.first <= self.cursor < self.first + self.visible:
            self.first = max(0, min(self.model.count - self.visible, self.cursor))
        self._sync_value()
        self.redraw()
        return "break"

    def zoom(self, factor):
        self.visible = max(1, min(self.model.count, int(self.visible * factor)))
        self.first = max(0, min(self.model.count - self.visible, self.cursor - self.visible // 2))
        self.redraw()

    def show_all(self):
        self.first, self.visible = 0, self.model.count
        self.redraw()

    def scroll(self, command, value, units=None):
        if command == "moveto":
            self.first = int(float(value) * self.model.count)
        else:
            self.first += int(value) * (self.visible if units == "pages" else max(1, self.visible // 10))
        self.first = max(0, min(self.model.count - self.visible, self.first))
        self.redraw()

    def redraw(self):
        if not self.winfo_exists():
            return
        first, last = self.model.selection
        for name, canvas in self.canvases.items():
            canvas.delete("all")
            width, height = canvas.winfo_width(), canvas.winfo_height()
            plot_width = max(1, width - self.LEFT - self.RIGHT)
            plot_height = max(1, height - self.TOP - self.BOTTOM)
            x = lambda i: self.LEFT + (i - self.first) * plot_width / self.visible
            left, right = max(first, self.first), min(last + 1, self.first + self.visible)
            if left < right:
                canvas.create_rectangle(x(left), self.TOP, x(right), height - self.BOTTOM,
                                        fill="#e2eddf", outline="")
            if name == "Wave":
                canvas.create_text(4, 12, text="Wave", anchor="w", fill="#333333")
                start = int(self.first * self.source_frame_ms * 8)
                stop = int((self.first + self.visible) * self.source_frame_ms * 8)
                chunk = self.source[start:stop]
                chunk = np.pad(chunk, (0, max(0, stop - start - len(chunk))))
                # Draw min/max per pixel so long recordings retain transients.
                if len(chunk):
                    for pixel in range(int(plot_width)):
                        a = int(pixel * len(chunk) / plot_width)
                        b = min(len(chunk), max(a + 1, int((pixel + 1) * len(chunk) / plot_width)))
                        values = chunk[a:b]
                        if len(values):
                            mid = self.TOP + plot_height / 2
                            canvas.create_line(self.LEFT + pixel, mid - float(values.max()) * plot_height / 2,
                                               self.LEFT + pixel, mid - float(values.min()) * plot_height / 2,
                                               fill="#777777")
            else:
                low, high = self._scale(name)
                for value in (low, (low + high) / 2, high):
                    y = self.TOP + (high - value) * plot_height / (high - low)
                    canvas.create_line(self.LEFT, y, width - self.RIGHT, y, fill="#e5e5e5")
                    canvas.create_text(self.LEFT - 8, y, text=f"{value:g}", anchor="e", fill="#666666")
                if name != "K":
                    canvas.create_text(4, 6, text=name, anchor="nw", fill="#333333")
                values = self.model.values(self._track_name(name))
                points = []
                for i in range(self.first, self.first + self.visible):
                    value = values[i]
                    if value is None:
                        if len(points) >= 4:
                            canvas.create_line(*points, fill="#254a86", width=1.5)
                        points = []
                        continue
                    y = self.TOP + (high - value) * plot_height / (high - low)
                    points.extend((x(i), y, x(i + 1), y))
                if len(points) >= 4:
                    canvas.create_line(*points, fill="#254a86", width=1.5)
            if self.first <= self.cursor < self.first + self.visible:
                canvas.create_line(x(self.cursor + 0.5), self.TOP, x(self.cursor + 0.5),
                                   height - self.BOTTOM, fill="#444444", dash=(2, 3))
            for tick in range(6):
                frame = self.first + tick * self.visible / 5
                canvas.create_text(x(frame), height - 9, text=f"{frame * self.frame_ms / 1000:.2f}",
                                   fill="#666666")
        self.scrollbar.set(self.first / self.model.count, (self.first + self.visible) / self.model.count)
        self.undo_button.configure(state="normal" if self.model.undo_stack else "disabled")
        self.redo_button.configure(state="normal" if self.model.redo_stack else "disabled")
        self.readout.set(f"Frame {self.cursor}   {self.cursor * self.frame_ms / 1000:.3f} s"
                         f"     {first}–{last}   {(last - first + 1) * self.frame_ms / 1000:.3f} s")
