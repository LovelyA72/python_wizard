### Updates to base python_wizard

- Added "lpcplayer" package (based on talkie) to support Play feature in GUI. 

- Added support for other LPC coding tables and a **Talkie** conversion. Talkie
  uses the TMS5220 frame format and base quantization, while GUI playback and
  synthesized preview WAVs use the software synthesis algorithm ported from the
  Arduino Talkie library. Normal Talkie conversion also runs a fast,
  source-aware burst guard. With QBP it checks voiced and unvoiced frames,
  reducing gain and, when needed, damping coefficients to control fixed-point
  buzzing. Other estimators use the existing noise-only guard. The LPC format,
  timing, and voicing are preserved. **Talkie mitigation
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

### Text to speech

The GUI's **Text to speech…** dialog includes Flite's US English SLT (female)
and RMS (male) Clustergen voices. Choose **TMS5100**, **TMS5110A**, **TMS5200**,
**TMS5220**, or **Talkie** for the preview and applied
LPC frames. **Play** synthesizes changed text and plays the chip preview.
The same button stops synthesis or playback. **Apply** synthesizes if needed
and replaces the current workspace, including any LPC edits. Closing the
dialog without applying leaves the workspace intact.

The dialog provides pitch (−12 to +12 semitones), formants (70–130%), speed
(0.5–2×), intonation (0–200%), and gain (−24 to +12 dB). Formants and pitch
are independent, and 0% intonation flattens the pitch contour. Speed changes
Flite's phoneme durations. Gain changes the encoded LPC energy, with output
headroom reserved to limit clipping. Pitch and energy remain limited by the
selected engine's tables. Settings are retained when reopening the dialog.

Build the native backend once before using the dialog:

```text
python setup.py
```

The build requires CMake, Git, and a C compiler. On Windows, MinGW-w64 GCC
and `mingw32-make` on PATH are supported. It downloads a pinned Flite source
revision and builds the helper with its voice into `pywizard/bin`. An existing
checkout can be supplied with `python setup.py build_flite --source PATH`.
Package builds also build and include the helper, voice registry, and Flite
license. WAV conversion, LPC editing, playback, and export work without the
helper. `PYWIZARD_FLITE_HELPER` can point
to an externally built copy of this helper. A stock `flite` executable does
not provide the required parameter output.

The backend captures smoothed F0 and mel-cepstral parameters before Flite's
waveform vocoder. It fits a ten-order LPC envelope over 0–4 kHz, quantizes
25 ms frames for the selected engine, and derives gain from its excitation
and filter response. Preemphasis controls DC amplification, and gain smoothing
avoids frame-to-frame volume pumping.
The helper preserves Flite's voicing decisions when an additional intonation
model supplies pitch over unvoiced consonants, so those frames use chip noise
excitation rather than a pitched chirp.
There is no intermediate synthesized speech waveform to analyze. Preview PCM
is generated only from the resulting LPC frames. Acoustic inference is cached
for unchanged text, voice, and speed, including when switching preview engines
or adjusting pitch, formants, intonation, and gain. PyAudio is
required for playback, but Apply and export work without it.

Voice definitions live in `pywizard/flite_voices.json`. Each entry supplies
an `id`, a display `label`, and the Flite source `model` name, such as
`cmu_us_rms`. To add another English Clustergen voice, add its entry and rebuild
the helper against a Flite checkout containing that model. CMake generates
the native registration table from the same registry used by the dialog.

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

GUI conversion runs in a separate process. Opening a WAV or changing settings
does not start conversion automatically. Click **Convert** to process the WAV.
Use **File > Export WAV** to save the current synthesis.
