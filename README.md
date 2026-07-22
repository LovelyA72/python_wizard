### Updates to base python_wizard

- Added "lpcplayer" package (based on talkie) to support Play feature in GUI. 

- Added dirty support for other LPC coding tables. TMS5100 is default now.
  ```
    -T {tms5220,tms5100}, --tablesVariant {tms5220,tms5100}
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

`Processor` uses the existing autocorrelation detector by default. The native
NumPy YIN detector can be selected from the command line with
`--pitchDetector yin`; its trough threshold defaults to `0.1` and can be set
with `--yinThreshold`. A different algorithm can also be supplied without
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

Autocorrelation LPC remains the default. Native NumPy Burg LPC can be selected
with `--lpcEstimator burg`. LPC estimators implement `LpcEstimator.estimate()`
and return reflection coefficients plus residual energy, and custom estimators
can be passed to `Processor` with its `lpc_estimator` argument.

Usage: 
```
       python_wizard [-h] [-u UNVOICEDTHRESHOLD] [-w WINDOWWIDTH] [-U] [-V]
                        [-S] [-p] [-a PREEMPHASISALPHA] [-d] [-r PITCHRANGE]
                        [-F FRAMERATE] [-m SUBMULTIPLETHRESHOLD]
                        [--pitchDetector {autocorrelation,yin}]
                        [--yinThreshold YINTHRESHOLD]
                        [--lpcEstimator {autocorrelation,burg}]
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
  --lpcEstimator {autocorrelation,burg}
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

The GUI exposes the same optimizer in an **Analysis-by-synthesis optimizer**
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

