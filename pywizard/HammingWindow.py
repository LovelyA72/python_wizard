import logging
import numpy as np

class HammingWindow(object):
    _windows = {}

    @classmethod
    def processBuffer(cls, buf):
        l = len(buf)
        if l not in cls._windows:
            logging.debug("HammingWindow: Generate window for len {}".format(l))
            cls._windows[l] = np.hamming(l)

        buf.samples *= cls._windows[l]

