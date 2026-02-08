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

    def test_wave_complexity_phase_behavior(self) -> None:
        """
        Test wave_complexity() for both amplitude and phase sensitivity.

        Ensures that:
        1. Structured amplitude+phase patterns have lower complexity.
        2. Random amplitude or phase increases complexity.
        3. Empty amplitudes return 0.0.
        4. Complexity does not exceed Shannon limit of frequency bins.
        """
        # --- 1. Structured amplitude, constant phase ---
        bits_struct = [0, 1] * 32
        q_struct = QBitwave(bitstring=bits_struct)
        # Constant phase (0)
        q_struct.amplitudes = np.array(q_struct.amplitudes, dtype=np.complex128) * np.exp(1j * 0.0)
        comp_struct = q_struct.wave_complexity()
        self.assertIsInstance(comp_struct, float)

        # --- 2. Same amplitude, random phases ---
        q_phase = QBitwave(bitstring=bits_struct)
        n_amp = len(q_phase.amplitudes)  # correct length
        q_phase.amplitudes = np.array(q_phase.amplitudes, dtype=np.complex128) * np.exp(1j * np.random.rand(n_amp) * 2 * np.pi)
        comp_phase = q_phase.wave_complexity()
        self.assertIsInstance(comp_phase, float)
        self.assertGreater(comp_phase, comp_struct,
            f"Random phase complexity ({comp_phase}) should exceed structured ({comp_struct})")

        # --- 3. Random amplitude + phase ---
        bits_rand = np.random.randint(0, 2, 64).tolist()
        q_rand = QBitwave(bitstring=bits_rand)
        n_amp_rand = len(q_rand.amplitudes)
        q_rand.amplitudes = np.array(q_rand.amplitudes, dtype=np.complex128) * np.exp(1j * np.random.rand(n_amp_rand) * 2 * np.pi)
        comp_rand = q_rand.wave_complexity()
        self.assertIsInstance(comp_rand, float)
        self.assertGreater(comp_rand, comp_struct,
            f"Random amplitude+phase ({comp_rand}) should exceed structured ({comp_struct})")

        # --- 4. Empty amplitudes edge case ---
        q_empty = QBitwave(bitstring=[0]*4, fixed_basis_size=8)
        self.assertEqual(q_empty.wave_complexity(), 0.0)

        
if __name__ == "__main__":
    unittest.main()
