import unittest

import numpy as np

from real_fault_lab import (
    hamming74_decode,
    hamming74_encode,
)


class RealFaultLabTests(unittest.TestCase):
    def test_hamming74_roundtrip(self):
        payload = bytes(range(256))
        encoded, length = hamming74_encode(payload)
        self.assertEqual(hamming74_decode(encoded, length), payload)

    def test_hamming74_corrects_one_bit_per_codeword(self):
        payload = b"real text payload"
        encoded, length = hamming74_encode(payload)
        damaged = encoded.copy().reshape(-1, 7)
        damaged[:, 3] ^= 1
        self.assertEqual(
            hamming74_decode(damaged.ravel(), length),
            payload,
        )

    def test_hamming74_detected_by_external_crc_after_double_error(self):
        payload = b"crc protected payload"
        encoded, length = hamming74_encode(payload)
        damaged = encoded.copy()
        damaged[0] ^= np.uint8(1)
        damaged[1] ^= np.uint8(1)
        self.assertNotEqual(
            hamming74_decode(damaged, length),
            payload,
        )


if __name__ == "__main__":
    unittest.main()

