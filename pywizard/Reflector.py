import numpy as np
from pywizard.userSettings import settings

class Reflector(object):
    """
    Implements the reflector parameter guessing for the LPC
    algorithm.

    Test if stop frames do not accidentally created
    >>> from CodingTable import CodingTable

    >>> ct = CodingTable()

    >>> r = Reflector()

    >>> r.rms = ct.rms[15]

    >>> r.limitRMS = True

    >>> r.rms == ct.rms[14]
    True
    """
    kNumberOfKParameters = 11

    def __init__(self, codingTable, k=None, rms=None, limitRMS=None):
        # TODO From config!!
        self.unvoicedThreshold = settings.unvoicedThreshold
        self.codingTable = codingTable
        if (k is None):
            assert(rms is None)
            assert(limitRMS is None)
            self.limitRMS = False
            self.ks = [0] * self.kNumberOfKParameters;
        else:
            assert(rms is not None)
            assert(limitRMS is not None)
            self._rms = rms
            self.ks = k
            self.limitRMS = limitRMS

    @classmethod
    def formattedRMS(cls, rms, numberOfSamples):
        return np.sqrt(rms / numberOfSamples) * (1 << 15)

    @classmethod
    def fromLpcEstimate(cls, codingTable, estimate, numberOfSamples):
        rms = cls.formattedRMS(estimate.residual_energy, numberOfSamples)
        return cls(
            codingTable,
            k=list(estimate.reflection_coefficients),
            rms=rms,
            limitRMS=True,
        )

    @classmethod
    def translateCoefficients(cls, codingTable, r, numberOfSamples):
        """Translate autocorrelation coefficients through the legacy API."""
        from pywizard.LpcEstimator import AutocorrelationLpcEstimator

        estimate = AutocorrelationLpcEstimator().estimate_from_autocorrelation(r)
        return cls.fromLpcEstimate(codingTable, estimate, numberOfSamples)


    @property
    def rms(self):
        if self.limitRMS and self._rms >= self.codingTable.rms[self.codingTable.kStopFrameIndex - 1]:
            return self.codingTable.rms[self.codingTable.kStopFrameIndex - 1]
        else:
            return self._rms

    @rms.setter
    def rms(self, rms):
        self._rms = rms

    def isVoiced(self):
        return not self.isUnvoiced()

    def isUnvoiced(self):
        return self.ks[1] >= self.unvoicedThreshold

