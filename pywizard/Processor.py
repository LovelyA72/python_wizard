from pywizard.Buffer import Buffer
from pywizard.Filterer import Filterer
from pywizard.Reflector import Reflector
from pywizard.Segmenter import Segmenter
from pywizard.PitchDetector import create_pitch_detector_from_settings
from pywizard.LpcEstimator import create_lpc_estimator_from_settings
from pywizard.userSettings import settings
from pywizard.HammingWindow import HammingWindow
from pywizard.FrameData import FrameData
from pywizard.PreEmphasizer import PreEmphasizer
from pywizard.CodingTable import CodingTable
import numpy as np
import logging

class Processor(object):
    def __init__(self, buf, model=None, pitch_detector=None, lpc_estimator=None):
        self.mainBuffer = buf
        self.pitchTable = None
        self.pitchBuffer = Buffer.copy(buf)
        self.codingTable = CodingTable(model)
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

        if settings.preEmphasis:
            PreEmphasizer.processBuffer(buf)

        self.pitchTable = {}
        wrappedPitch = False
        if settings.overridePitch:
            wrappedPitch = settings.pitchValue
        else:
            self.pitchTable = self.pitchTableForBuffer(self.pitchBuffer)

        segmenter = Segmenter(buf=self.mainBuffer, windowWidth=settings.windowWidth)

        frames = []
        for (cur_buf, i) in segmenter.eachSegment():
            HammingWindow.processBuffer(cur_buf)
            lpc_estimate = self.lpcEstimator.estimate(cur_buf)
            reflector = Reflector.fromLpcEstimate(
                self.codingTable,
                lpc_estimate,
                cur_buf.size,
            )

            if wrappedPitch:
                pitch = int(wrappedPitch)
            else:
                pitch = self.pitchTable[i]

            frameData = FrameData(reflector, pitch, repeat=False)

            frames.append(frameData)

        if settings.includeExplicitStopFrame:
            frames.append(frameData.stopFrame())

        self.frames = frames

    def pitchTableForBuffer(self, pitchBuffer):
        filterer = Filterer(pitchBuffer, lowPassCutoffInHZ=settings.minimumPitchInHZ, highPassCutoffInHZ=settings.maximumPitchInHZ, gain=1)
        buf = filterer.process()

        segmenter = Segmenter(buf, windowWidth=2)
        pitchTable = np.zeros(segmenter.numberOfSegments())

        for (buf, index) in segmenter.eachSegment():
            pitchTable[index] = self.pitchDetector.detect(buf)

        return pitchTable


    def process(self):
        return(self.frameData)

