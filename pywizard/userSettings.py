import logging
from collections import OrderedDict

class userSettings(object):
    availableSettings = ["pitchValue", "unvoicedThreshold", "windowWidth",
        "normalizeUnvoicedRMS", "normalizeUnvoicedRMS", "includeExplicitStopFrame",
        "preEmphasis", "preEmphasisAlpha", "overridePitch", "pitchOffset",
        "minimumPitchInHZ", "maximumPitchInHZ", "frameRate",
        "pitchDetector", "subMultipleThreshold", "yinThreshold",
        "lpcEstimator", "outputFormat", "rmsLimit", "tablesVariant",
        "talkieMitigationStrength", "talkieGateThreshold"]
    pitchValue = 0
    unvoicedThreshold = 0.7
    windowWidth = 2
    normalizeUnvoicedRMS = False
    normalizeVoicedRMS = False
    includeExplicitStopFrame = True
    preEmphasis = True
    preEmphasisAlpha = -0.9373
    overridePitch = False
    pitchOffset = 0
    maximumPitchInHZ = 500
    minimumPitchInHZ = 50
    frameRate = 25
    pitchDetector = "yin"
    subMultipleThreshold = 0.9
    yinThreshold = 0.9
    lpcEstimator = "autocorrelation"
    outputFormat = "arduino"
    rmsLimit = 14
    tablesVariant = "tms5100"
    talkieMitigationStrength = 1.5
    talkieGateThreshold = 0.005

    def import_from_argparse(self, raw):
        v = vars(raw)
        self.import_from_dict(v)

    def import_from_dict(self, input_dict):
        error_list = []
        for key in input_dict:
            if key=='pitchRange':
                (self.minimumPitchInHZ, self.maximumPitchInHZ) = [ int(x) for x in input_dict[key].split(",") ]
            else:
                try:
                    self.__setattr__(key, type(self.__getattribute__(key))(input_dict[key]))
                except AttributeError:
                    logging.debug("Discarding argument {}={}".format(key, input_dict[key]))
                except ValueError:
                    error_list.append(key)
        if len(error_list) > 0:
            return error_list
        else:
            return None

    def export_to_odict(self):
        r = OrderedDict()
        for k in self.availableSettings:
            r[k] = self.__getattribute__(k)
        return r


settings = userSettings()

