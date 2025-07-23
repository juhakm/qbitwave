import unittest
import numpy as np
from qbitwave.qbitwave import QBitwave 

class TestQBitwave(unittest.TestCase):
    """
    Unit tests for the QBitwave class.
    """

    def test_initialization_and_structure(self) -> None:
        """
        Test that the object initializes and block size is selected.
        """
        q = QBitwave("01011001100100101110101010101010")
        self.assertIsInstance(q.get_amplitudes(), list)
        self.assertGreater(q.num_states(), 0)
        self.assertIsNotNone(q.get_selected_block_size())

    def test_entropy_nonzero(self) -> None:
        """
        Test that entropy is computed and is nonzero for a meaningful bitstring.
        """
        q = QBitwave("01010101101010110101011100001111")
        entropy: float = q.entropy()
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, np.log2(q.dimension()))

    def test_amplitude_normalization(self) -> None:
        """
        Ensure the L2 norm of the wavefunction is approximately 1.
        """
        q = QBitwave("11110000111100001111000011110000")
        norm: float = q.norm()
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_probability_distribution_sum(self) -> None:
        """
        Test that the probability distribution sums to ~1.
        """
        q = QBitwave("00110011001100110011001100110011")
        probs: np.ndarray = q.get_probability_distribution()
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_phase_distribution_shape(self) -> None:
        """
        Ensure phase distribution has same length as number of states.
        """
        q = QBitwave("10101010101010101010101010101010")
        phases: np.ndarray = q.get_phase_distribution()
        self.assertEqual(len(phases), q.dimension())

    def test_str_output(self) -> None:
        """
        Check that __str__ returns a Dirac-style wavefunction string.
        """
        q = QBitwave("10110100101101001011010010110100")
        wf_str: str = str(q)
        self.assertIn("|", wf_str)  # check Dirac notation is used
        self.assertIsInstance(wf_str, str)

    def test_empty_amplitudes_for_short_input(self) -> None:
        """
        Test behavior for bitstrings too short to yield amplitudes.
        """
        q = QBitwave("10")  # too short for min_block_size=4
        self.assertEqual(q.get_amplitudes(), [])


    def test_rejects_zero_norm(self) -> None:
        """
        Check that all-zero wavefunctions are rejected.
        """
        # 16 zero bits would translate to real=0, imag=0 repeatedly
        q = QBitwave("0" * 64)
        self.assertEqual(q.get_amplitudes(), [])


if __name__ == '__main__':
    unittest.main()
