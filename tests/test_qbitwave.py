import unittest
import numpy as np
from qbitwave import QBitwave


class TestQBitwave(unittest.TestCase):

    def test_initialization(self):
        bw = QBitwave("010101")
        self.assertIsInstance(bw.bitstring, str)
        self.assertTrue(all(b in "01" for b in bw.bitstring))

    def test_amplitudes_length(self):
        bw = QBitwave("1100110011")
        amplitudes = bw.get_amplitudes()
        self.assertIsInstance(amplitudes, np.ndarray)
        self.assertGreaterEqual(len(amplitudes), 1)
        self.assertTrue(np.iscomplexobj(amplitudes))

    def test_amplitudes_normalization(self):
        bw = QBitwave("1100110011")
        amplitudes = bw.get_amplitudes()
        prob_sum = np.sum(np.abs(amplitudes) ** 2)
        self.assertAlmostEqual(prob_sum, 1.0, places=6)

    def test_entropy_increases_with_structure(self):
        bw_low = QBitwave("0000000000")
        bw_high = QBitwave("1011001010")
        entropy_low = bw_low.entropy()
        entropy_high = bw_high.entropy()
        self.assertLess(entropy_low, entropy_high)

    def test_resolution_consistency(self):
        bw = QBitwave("1011010110110101")
        self.assertEqual(bw.resolution(), len(bw.get_amplitudes()))

    def test_entropy_bounds(self):
        bw = QBitwave("00000000")
        entropy = bw.entropy()
        self.assertGreaterEqual(entropy, 0)
        self.assertLessEqual(entropy, np.log2(bw.resolution() + 1e-10))  # entropy ≤ log2(n)

    def test_num_states(self):
        bw = QBitwave("1010011101")
        n = bw.num_states()
        self.assertEqual(n, len(bw.get_amplitudes()))
        self.assertGreater(n, 0)

if __name__ == '__main__':
    unittest.main()
