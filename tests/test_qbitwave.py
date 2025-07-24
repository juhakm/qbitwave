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

    def test_empty_amplitudes_for_short_input(self):
        # The block_size is 4, so 4 bits are required per amplitude
        # With only 3 bits, decoding should produce no amplitudes
        bits = [1, 0, 1]  # not enough for one block
        q = QBitwave(bits, fixed_block_size=4)
        self.assertEqual(list(q.get_amplitudes()), [])

    @unittest.skip("#FIX: Temporarily skipping this test")
    def test_rejects_zero_norm(self):
        # All-zero input bitstring, which should result in all amplitudes being 0+0j
        zero_input = [0] * 32  # or however many bits are needed to yield multiple amplitude blocks
        q = QBitwave(zero_input)

        amplitudes = list(q.get_amplitudes())

        # Check that all amplitudes are very close to 0, indicating zero norm
        norm = sum(abs(a) ** 2 for a in amplitudes)

        # If norm is effectively zero, wavefunction should have been discarded
        if norm < 1e-6:
            self.assertEqual(amplitudes, [], f"Expected empty amplitude list for zero-norm input, got: {amplitudes}")
        else:
            self.fail(f"Input did not yield zero norm as expected. Norm was {norm}, amplitudes: {amplitudes}")

    def test_mutate_changes_bits_and_recomputes_wavefunction(self):
        # Start with a known bitstring that produces amplitudes
        original_bits = [0, 1] * 32  # alternating bits
        q = QBitwave(original_bits)
        original_wavefunction = q.amplitudes.copy()
        original_length = len(original_wavefunction)

        q.mutate(mutation_rate=1.0)  # force full mutation

        # Ensure bitstring changed
        self.assertNotEqual(q.bitstring, original_bits)

        # Ensure new wavefunction is non-empty
        self.assertGreater(len(q.amplitudes), 0)

        # If the new wavefunction has same length, it should differ
        if len(q.amplitudes) == original_length:
            self.assertFalse(np.allclose(q.amplitudes, original_wavefunction))

        # Ensure it's still normalized
        norm = np.linalg.norm(q.amplitudes)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_flip_changes_one_bit_and_recomputes_wavefunction(self):
        bitstring = [0, 1] * 8  # 16 bits
        q = QBitwave(bitstring)

        original_bits = q.bitstring.copy()
        original_wavefunction = np.array(q.amplitudes)

        q.flip()

        # Confirm one bit was flipped
        diff_count = sum(b1 != b2 for b1, b2 in zip(original_bits, q.bitstring))
        self.assertEqual(diff_count, 1, "Exactly one bit should be flipped")

        # Confirm wavefunction still exists
        self.assertGreater(len(q.amplitudes), 0, "Wavefunction should not be empty")

        # Confirm change, if possible
        if len(original_wavefunction) == len(q.amplitudes):
            self.assertFalse(np.allclose(q.amplitudes, original_wavefunction),
                            "Wavefunction should change after flip")



if __name__ == '__main__':
    unittest.main()
