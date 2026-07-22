import os
import tempfile
import unittest

from pywizard.FrameDataBinaryEncoder import BitPacker


class _CodingTable:
    bits = (4,)
    kStopFrameIndex = 15

    @staticmethod
    def parameters():
        return ("gain",)


class _Frame:
    def __init__(self, gain):
        self.gain = gain

    def parameters(self):
        return {"gain": self.gain}


class _Processor:
    codingTable = _CodingTable()

    def __init__(self, gains):
        self.frames = [_Frame(gain) for gain in gains]


class _VerboseCodingTable:
    bits = (4, 4)
    kStopFrameIndex = 15

    @staticmethod
    def parameters():
        return ("gain", "unused")


class _VerboseFrame:
    @staticmethod
    def parameters():
        return {"gain": 15, "unused": 7}


class _VerboseProcessor:
    codingTable = _VerboseCodingTable()
    frames = [_VerboseFrame(), _VerboseFrame()]


class LpcExportTests(unittest.TestCase):
    def test_lpc_bytes_adds_stop_frame(self):
        self.assertEqual(BitPacker.lpc_bytes(_Processor([1])), bytes([0xF8]))

    def test_lpc_bytes_does_not_duplicate_existing_stop_frame(self):
        self.assertEqual(BitPacker.lpc_bytes(_Processor([15])), bytes([0x0F]))

    def test_lpc_bytes_discards_fields_and_frames_after_first_stop(self):
        self.assertEqual(BitPacker.lpc_bytes(_VerboseProcessor()), bytes([0x0F]))

    def test_raw_stream_preserves_partial_final_byte(self):
        self.assertEqual(BitPacker.raw_stream(_Processor([1])), [0x08])

    def test_write_lpc_writes_binary_data(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "speech.lpc")
            byte_count = BitPacker.write_lpc(_Processor([1]), filename)

            self.assertEqual(byte_count, 1)
            with open(filename, "rb") as input_file:
                self.assertEqual(input_file.read(), bytes([0xF8]))


if __name__ == "__main__":
    unittest.main()
