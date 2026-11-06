from pywizard.Buffer import Buffer
from pywizard.PitchDetector import create_pitch_detector_from_settings
from pywizard.LpcEstimator import LpcAnalysisContext, create_lpc_estimator_from_settings
from pywizard.userSettings import settings
from pywizard.QuantizedFrame import QuantizedFrame
from pywizard.PreEmphasizer import PreEmphasizer
from pywizard.CodingTable import CodingTable
import numpy as np

class Processor(object):
    def __init__(self, buf, model=None, pitch_detector=None, lpc_estimator=None):
        samples = np.asarray(buf.samples[buf.start:buf.end], dtype=float)
        if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
            raise ValueError("LPC analysis requires nonempty, finite mono audio")
        if not np.isfinite(buf.sampleRate) or buf.sampleRate <= 0:
            raise ValueError("sample rate must be positive")
        if not np.isfinite(settings.frameRate) or settings.frameRate <= 0:
            raise ValueError("frame duration must be positive")
        if not np.isfinite(settings.windowWidth) or settings.windowWidth <= 0:
            raise ValueError("window width must be positive")
        self.mainBuffer = Buffer(samples=samples.copy(), sampleRate=buf.sampleRate)
        self.pitchBuffer = Buffer(samples=samples.copy(), sampleRate=buf.sampleRate)
        self.codingTable = CodingTable(model or settings.tablesVariant)
        self.pitchDetector = (
            pitch_detector
            if pitch_detector is not None
            else create_pitch_detector_from_settings()
        )
        self.lpcEstimator = (
            lpc_estimator
            if lpc_estimator is not None
            else create_lpc_estimator_from_settings()
        )

        if settings.preEmphasis and self.mainBuffer.energy() > 0:
            PreEmphasizer.processBuffer(self.mainBuffer)
        self.analysisContext = LpcAnalysisContext(
            source=self.pitchBuffer, signal=self.mainBuffer,
            coding_table=self.codingTable, pitch_detector=self.pitchDetector,
            frame_samples=int(np.ceil(buf.sampleRate * settings.frameRate / 1000)),
            window_width=settings.windowWidth,
            minimum_pitch_hz=settings.minimumPitchInHZ,
            maximum_pitch_hz=settings.maximumPitchInHZ,
            override_pitch=float(settings.pitchValue) if settings.overridePitch else None,
            pitch_offset=int(settings.pitchOffset),
        )
        analysis = self.lpcEstimator.analyze(self.analysisContext)
        self.pitchTable = analysis.pitch_periods
        self.frames = list(analysis.frames)
        if settings.includeExplicitStopFrame:
            if self.frames and not isinstance(self.frames[-1], QuantizedFrame):
                self.frames.append(self.frames[-1].stopFrame())
            else:
                self.frames.append(QuantizedFrame(energy_index=15))

    def pitchTableForBuffer(self, pitchBuffer):
        from pywizard.Filterer import Filterer
        from pywizard.Segmenter import Segmenter
        filterer = Filterer(pitchBuffer, lowPassCutoffInHZ=settings.minimumPitchInHZ, highPassCutoffInHZ=settings.maximumPitchInHZ, gain=1)
        buf = filterer.process()

        segmenter = Segmenter(buf, windowWidth=2)
        pitchTable = np.zeros(segmenter.numberOfSegments())

        for (buf, index) in segmenter.eachSegment():
            pitchTable[index] = self.pitchDetector.detect(buf)

        return pitchTable


    def process(self):
        return self.frames

