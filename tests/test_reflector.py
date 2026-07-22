import unittest

from pywizard.Reflector import Reflector


class _CodingTable:
    kStopFrameIndex = 3
    rms = (0.0, 10.0, 20.0, 30.0)


class ReflectorTests(unittest.TestCase):
    def test_limited_rms_is_clamped_before_stop_frame(self):
        reflector = Reflector(
            _CodingTable(),
            k=[0.0] * Reflector.kNumberOfKParameters,
            rms=30.0,
            limitRMS=True,
        )

        self.assertEqual(reflector.rms, 20.0)


if __name__ == "__main__":
    unittest.main()
