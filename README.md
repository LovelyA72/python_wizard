### Updates to base python_wizard

- Added "lpcplayer" package (based on talkie) to support Play feature in GUI. 

- Added support for other LPC coding tables and a **Talkie** conversion. Talkie
  uses the TMS5220 frame format and base quantization, while GUI playback and
  synthesized preview WAVs use the software synthesis algorithm ported from the
  Arduino Talkie library. Normal Talkie conversion also runs a fast,
  source-aware burst guard. With QBP it checks voiced and unvoiced frames,
  reducing gain and, when needed, damping coefficients to control fixed-point
  buzzing. Other estimators use the existing noise-only guard. The LPC format,
  timing, and voicing are preserved; the full optimizer is not required. **Talkie mitigation
  strength** defaults to `2.0`, accepts `0.0` through `3.0`, and can be set to
  zero to disable the burst guard. **Talkie gate threshold** defaults to `0.01`
  and accepts `0.0` through `1.0` (zero disables it). Runs of at least three frames
  whose source RMS is below the threshold are encoded as silence. When a rest
  frame would retain a buzzing fixed-point filter state, the first gated frame
  is instead a zero-excitation transition that loads stable coefficients; the
  remaining rest frames then synthesize as exact silence.
  ```
    -T {tms5100,tms5110a,tms5200,tms5220,Talkie}
                          Tables variant
  ```                        

- Support for python formatted output
  ```
    -f {arduino,C,hex,python}, --outputFormat {arduino,C,hex,python}
                          Output file format
  ```
- Export raw, headerless `.lpc` byte streams from both the CLI and GUI. LPC
  exports always include an explicit stop frame and are directly compatible
  with Talkie and the FM720 Z80 file player.

- Small fixes and enhancements in GUI



------
This project is a python port of the great macOS tool BlueWizard (https://github.com/patrick99e99/BlueWizard), which is written in objective C and I was not familiar enough with this C dialect to make an portable command line application out of it.

It is intended to convert (voice) audio streams into LPC bitstreams used in the TMS 5220 chip or e.g. in the Arduino library Talkie. Now you can generate your own LPC streams and make your chips say the things you want them to.

Compared to BlueWizard some minor features have been added:
1. Ability to downsample a wave file automatically
2. Automated output formatters for C, Arduino (C-Dialect) and plain hex

Prerequisites:
- Python >= 3.13
- NumPy >= 2.1
- SciPy >= 1.14.1

Install the project and its command-line tools with:

```
python -m pip install .
```

### Pitch detector plugins

`Processor` uses the native NumPy YIN detector by default, with trough
threshold `0.7`. Select `--pitchDetector autocorrelation` for the earlier
detector or set the YIN threshold with `--yinThreshold`. A different algorithm can also be supplied without
changing the processing pipeline by implementing `PitchDetector.detect()` and
passing the detector to `Processor`:

```python
from pywizard.PitchDetector import PitchDetector
from pywizard.Processor import Processor


class MyPitchDetector(PitchDetector):
    def detect(self, buf):
        # Return the fundamental period in samples, or 0.0 if none is found.
        return period


processor = Processor(audio_buffer, pitch_detector=MyPitchDetector())
```

### LPC estimator plugins

QBP with Talkie is the application default. Autocorrelation LPC remains
available with `--lpcEstimator autocorrelation`. Native NumPy Burg LPC can be selected
with `--lpcEstimator burg`. LPC estimators implement `LpcEstimator.estimate()`
and return reflection coefficients plus residual energy, and custom estimators
can be passed to `Processor` with its `lpc_estimator` argument.

`--lpcEstimator qbp` selects a QBOX-informed sequence estimator. It uses
separate voiced/unvoiced windows, center clipping before the selected pitch
detector, temporal pitch/voicing repair, and gain estimated from quantized
coefficients and the target's excitation. It supports the existing chip and
Talkie targets; it does not load QBOX or its D6 format. This is a new
implementation, not a bit-exact QBOX port, and audible quality should be
compared on your recordings.

CLI and GUI share the QBP/Talkie defaults: YIN threshold `0.7`, preemphasis
enabled, Talkie mitigation strength `2.0`, and gate threshold `0.01`. For QBP,
the burst guard checks voiced frames as well as noise frames. It first lowers
gain, then damps quantized coefficients only when a retained filter state
continues to exceed source-relative limits. Other estimators retain their
noise-only guard. Use `--no-preEmphasis` to disable preemphasis from the CLI.

```text
python python_wizard --lpcEstimator qbp --pitchDetector yin -T Talkie -f lpc speech.wav
```

Existing `estimate(buf)` plugins still work. A sequence-aware plugin overrides
`analyze(context)` and returns `LpcAnalysis(frames, pitch_periods)`. The context
contains original PCM, preemphasized PCM, coding tables, pitch detector, frame
size, window width, and pitch controls. Its frames can be `FrameData` or
`QuantizedFrame` objects; `Processor` adds the optional stop frame. Input PCM
is copied before analysis.

After conversion, **Edit LPC** opens waveform, gain, pitch, and K1–K10 tracks.
Select a frame range, draw parameter curves, or set a numeric value. Gain uses
table indices 0–14, pitch uses Hz, and K values use normalized coefficients.
K5–K10 are unavailable on unvoiced frames. Edits update the current conversion
and its exports; each draw gesture or range edit supports Undo/Redo. Repeated
frames are expanded when editing to preserve neighboring coefficients.
**Play selection** decodes preceding frames before playing the range, retaining
the synthesizer state. Playback requires PyAudio; editing and export do not.

Run the regression checks with `python -m unittest discover -s tests`.

Usage: 
```
       python_wizard [-h] [-u UNVOICEDTHRESHOLD] [-w WINDOWWIDTH] [-U] [-V]
                        [-S] [-p] [-a PREEMPHASISALPHA] [-d] [-r PITCHRANGE]
                        [-F FRAMERATE] [-m SUBMULTIPLETHRESHOLD]
                        [--pitchDetector {autocorrelation,yin}]
                        [--yinThreshold YINTHRESHOLD]
                        [--lpcEstimator {autocorrelation,burg,qbp}]
                        [-f {arduino,C,hex,python,lpc}] [-o OUTPUT]
                        filename

positional arguments:
  filename              File name of a .wav file to be processed

optional arguments:
  -h, --help            show this help message and exit
  -u UNVOICEDTHRESHOLD, --unvoicedThreshold UNVOICEDTHRESHOLD
                        Unvoiced frame threshold
  -w WINDOWWIDTH, --windowWidth WINDOWWIDTH
                        Window width in frames
  -U, --normalizeUnvoicedRMS
                        Normalize unvoiced frame RMS
  -V, --normalizeVoicedRMS
                        Normalize voiced frame RMS
  -S, --includeExplicitStopFrame
                        Create explicit stop frame (needed e.g. for Talkie)
  -p, --preEmphasis     Pre emphasize sound to improve quality of translation
  -a PREEMPHASISALPHA, --preEmphasisAlpha PREEMPHASISALPHA
                        Emphasis coefficient
  -d, --debug           Enable (lots) of debug output
  -r PITCHRANGE, --pitchRange PITCHRANGE
                        Comma separated range of available voice pitch in Hz.
                        Default: 50,500
  -F FRAMERATE, --frameRate FRAMERATE
  -m SUBMULTIPLETHRESHOLD, --subMultipleThreshold SUBMULTIPLETHRESHOLD
                        Autocorrelation sub-multiple threshold
  --pitchDetector {autocorrelation,yin}
                        Pitch detection algorithm
  --yinThreshold YINTHRESHOLD
                        YIN trough threshold
  --lpcEstimator {autocorrelation,burg,qbp}
                        LPC coefficient estimation algorithm
  -f {arduino,C,hex,python,lpc}, --outputFormat {arduino,C,hex,python,lpc}
                        Output file format
  -o OUTPUT, --output OUTPUT
                        Write output to this file
```

Export a binary LPC file with an explicit stop frame:

```text
python_wizard -f lpc -o SPEECH.LPC speech.wav
```

If `-o` is omitted for LPC output, the input name is reused with a `.lpc`
extension. In the GUI, process a WAV file and select **File > Export LPC...**.

### Analysis-by-synthesis optimization

The command-line encoder can optionally improve quantized TMS52xx table
indices with deterministic chip synthesis and bounded coordinate descent. The
default path and bitstream writer are unchanged when `--optimize` is omitted.

```text
python_wizard --optimize --optimize-passes 2 --optimize-radius 1 \
  --optimize-loss-profile balanced -f lpc -o speech.lpc speech.wav
```

For listening tests and regression analysis, export both synthesized versions
and a detailed report:

```text
python_wizard --optimize --optimize-report optimization.json \
  --optimize-initial-wav initial.wav \
  --optimize-optimized-wav optimized.wav \
  --optimize-parameters frames.json -f lpc -o speech.lpc speech.wav
```

The optimizer preserves silence, stop, repeat, and voiced/unvoiced structure.
It searches energy, pitch, and individual K table neighbors in context. It is
an offline quality pass, so it is slower than normal encoding; phase-sensitive
waveform error is deliberately balanced with spectral and energy-envelope
terms. Voicing changes and exhaustive/global search are not currently done.

The GUI exposes the same optimizer in an **Analysis-by-synthesis**
panel. Open a WAV, select the pass count, search radius, lookahead, and loss
profile, then click **Convert**. Conversion is deliberately manual: opening a
file or changing settings does not start work automatically. Analysis and
optimization run in a spawned worker process with a separate Python interpreter,
while an ASCII progress bar in the status line reports progress without
freezing the interface. **Play LPC** becomes
available after conversion completes. Initial and optimized synthesis WAVs,
the JSON report, and indexed frame parameters can be saved from the **File**
menu. Finite-choice encoder settings—including Vocoder/chip, Output
language/format, Pitch estimator, and LPC estimator—are read-only dropdowns;
numeric settings remain editable fields.

