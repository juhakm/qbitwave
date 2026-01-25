import unittest
import numpy as np
from qbitwave.qbitwave import QBitwave
from typing import List


class TestQBitwave(unittest.TestCase):
    """
    Unit tests for the QBitwave class.

    Tests cover:
    - Initialization from bitstring and amplitudes
    - Entropy and compressibility calculations
    - Wavefunction normalization
    - Bitstring mutations and flips
    - Handling of edge cases (short or zero bitstrings)
    """

    def test_initialization_and_structure(self) -> None:
        """
        Test that the object initializes correctly and selects a basis size.
        """
        bits = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        q = QBitwave(bitstring=bits, fixed_basis_size=4)
        self.assertIsInstance(q.amplitudes, np.ndarray)
        self.assertGreater(len(q.amplitudes), 0)
        self.assertIsNotNone(q.selected_basis_size)

    def test_entropy_nonzero(self) -> None:
        """
        Test that wavefunction entropy is computed and nonzero for structured bitstring.
        """
        bits = [0, 1] * 16
        q = QBitwave(bitstring=bits)
        entropy = q.entropy()
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, np.log2(len(q.amplitudes)))

    def test_amplitude_normalization(self) -> None:
        """
        Ensure the L2 norm of the wavefunction is approximately 1.
        """
        bits = [1, 0] * 16
        q = QBitwave(bitstring=bits)
        norm = np.linalg.norm(q.amplitudes)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_compressibility_range(self) -> None:
        """
        Compressibility should be in [0, 1].
        """
        bits = [0, 1] * 16
        q = QBitwave(bitstring=bits)
        comp = q.compressibility()
        self.assertGreaterEqual(comp, 0.0)
        self.assertLessEqual(comp, 1.0)

    def test_bit_entropy(self) -> None:
        """
        Bit-level entropy should reflect bitstring randomness.
        """
        bits = [0, 1] * 16
        q = QBitwave(bitstring=bits)
        bit_entropy = q.bit_entropy()
        self.assertGreaterEqual(bit_entropy, 0.0)
        self.assertLessEqual(bit_entropy, 1.0)

    def test_mutate_changes_bits_and_wavefunction(self) -> None:
        """
        Test that mutate() alters bitstring and recomputes amplitudes.
        """
        bits = [0, 1] * 16
        q = QBitwave(bitstring=bits)
        original_bits = q.bitstring.copy()
        original_amplitudes = q.amplitudes.copy()

        q.mutate(0.5)

        # Bitstring should change
        self.assertNotEqual(q.amplitudes.tolist(), original_amplitudes.tolist())
        # Wavefunction should still exist and normalized
        self.assertGreater(len(q.amplitudes), 0)
        self.assertAlmostEqual(np.linalg.norm(q.amplitudes), 1.0, places=5)

    def test_flip_changes_one_bit(self) -> None:
        """
        Test that flip() toggles exactly one bit and updates wavefunction.
        """
        bits = [0, 1] * 8
        q = QBitwave(bitstring=bits)
        original_bits = q.bitstring.copy()
        original_amplitudes = q.amplitudes.copy()

        q.flip()

        diff_count = sum(b1 != b2 for b1, b2 in zip(original_bits, q.bitstring))
        self.assertEqual(diff_count, 1)
        self.assertGreater(len(q.amplitudes), 0)
        if len(original_amplitudes) == len(q.amplitudes):
            self.assertFalse(np.allclose(q.amplitudes, original_amplitudes))

    def test_zero_bitstring_produces_empty_amplitudes(self) -> None:
        """
        Ensure that a too-short or all-zero bitstring produces empty amplitude array.
        """
        bits = [0] * 3
        q = QBitwave(bitstring=bits, fixed_basis_size=4)
        self.assertEqual(len(q.amplitudes), 0)

    def test_coherence_metric_nonnegative(self) -> None:
        """
        Test that coherence() returns a non-negative float.
        """
        bits = [0, 1] * 16
        q = QBitwave(bitstring=bits)
        coh = q.coherence()
        self.assertGreaterEqual(coh, 0.0)

    def test_wave_complexity_entropy_behavior(self) -> None:
        """
        Test that wave_complexity() returns spectral entropy (float)
        and correctly identifies that random noise is more complex 
        than structured patterns.
        """
        # 1. Structured bitstring (low entropy/complexity)
        # [0, 1] repeated is a very simple, predictable wave.
        bits_structured = [0, 1] * 32
        q_struct = QBitwave(bitstring=bits_structured)
        comp_struct = q_struct.wave_complexity()
        
        self.assertIsInstance(comp_struct, float)
        
        # 2. Random bitstring (high entropy/complexity)
        # Random bits should always yield a higher entropy than a pure cycle.
        bits_random = np.random.randint(0, 2, 64).tolist()
        q_rand = QBitwave(bitstring=bits_random)
        comp_rand = q_rand.wave_complexity()
        
        self.assertIsInstance(comp_rand, float)
        
        # 3. Validation: Structure < Random
        self.assertLess(comp_struct, comp_rand, 
            f"Structured complexity ({comp_struct}) should be less than random ({comp_rand})")

        # 4. Edge case: empty amplitudes
        q_empty = QBitwave(bitstring=[0]*4, fixed_basis_size=8)
        self.assertEqual(q_empty.wave_complexity(), 0.0)

        # 5. Shannon Limit check
        # Complexity should never exceed log2 of the number of available frequency bins
        h_max = np.log2(len(q_rand.amplitudes)) + 0.01 # allow tiny epsilon
        self.assertLessEqual(comp_rand, h_max)
        
if __name__ == "__main__":
    unittest.main()
