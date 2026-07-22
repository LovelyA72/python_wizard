from pywizard.tools import BitHelpers
from pywizard.HexConverter import HexConverter
from collections import namedtuple
import logging

class BitPacker(object):

    @classmethod
    def _nibbles(cls, processor, ensure_stop=False):
        parametersList = [x.parameters() for x in processor.frames]
        if ensure_stop:
            gain_parameter = processor.codingTable.parameters()[0]
            stop_gain = processor.codingTable.kStopFrameIndex
            for index, parameters in enumerate(parametersList):
                if parameters.get(gain_parameter) == stop_gain:
                    parametersList = parametersList[:index] + [{gain_parameter: stop_gain}]
                    break
            else:
                parametersList.append({gain_parameter: stop_gain})
        return FrameDataBinaryEncoder.process(processor.codingTable, parametersList)

    @classmethod
    def raw_stream(cls, processor):
        raw_data = cls._nibbles(processor)
        return HexConverter.preprocess(raw_data)

    @classmethod
    def lpc_bytes(cls, processor):
        raw_data = cls._nibbles(processor, ensure_stop=True)
        return bytes(HexConverter.preprocess(raw_data))

    @classmethod
    def write_lpc(cls, processor, filename):
        data = cls.lpc_bytes(processor)
        with open(filename, "wb") as output_file:
            output_file.write(data)
        return len(data)

    @classmethod
    def pack(cls, processor):
        raw_data = cls._nibbles(processor)
        return HexConverter.process(raw_data)

class FrameDataBinaryEncoder(object):
    @classmethod
    def process(cls, codingTable, parametersList):
        bits = codingTable.bits
        binary = ""
        for parameters in parametersList:
            params = codingTable.parameters()
            for (param_name, idx) in zip(params, range(len(params))):
                if param_name not in parameters:
                    break
                value = parameters[param_name]
                binaryValue = BitHelpers.valueToBinary(value, bits[idx])
                logging.debug("param_name={} idx={} value={} binaryValue={}".format(param_name, idx, value, binaryValue))
                binary += binaryValue
        return cls.nibblesFrom(binary)

    @classmethod
    def nibblesFrom(cls, binary):
        # LPC frames are bit-packed rather than byte-aligned. Preserve the
        # final partial byte by padding unused high bits with zeroes.
        binary += "0" * ((8 - len(binary) % 8) % 8)
        nibbles = []
        while len(binary) >= 4:
            nibble = binary[0:4]
            binary = binary[4:]
            nibbles.append(nibble)
        return nibbles
