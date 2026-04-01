import unittest
import numpy as np
from qbitwave.qbitspinor import QBitSpinor


class TestQBitSpinor(unittest.TestCase):

    def setUp(self):
        """Initialize a standard spinor model for testing."""
        self.N = 10
        self.model = QBitSpinor(self.N)

    # -- Initialization and Mode Management --

    def test_initialization(self):
        """Verify domain size and empty state."""
        self.assertEqual(self.model.N, 10)
        self.assertEqual(len(self.model.modes), 0)

    def test_mode_storage(self):
        """Ensure modes store both spinor components."""
        self.model.modes[2] = (1+0j, 0+1j)
        self.assertIn(2, self.model.modes)
        a, b = self.model.modes[2]
        self.assertTrue(np.iscomplexobj(a))
        self.assertTrue(np.iscomplexobj(b))

    def test_clear_modes(self):
        """Ensure modes are cleared."""
        self.model.modes[1] = (1+0j, 1+0j)
        self.model.clear_modes()
        self.assertEqual(len(self.model.modes), 0)

    # -- Encoding --

    def test_encode_spinor_signal_basic(self):
        """Encoding should populate modes."""
        alpha = np.array([1+0j, -1+0j, 1+0j, -1+0j])
        beta  = np.array([0+1j, 0-1j, 0+1j, 0-1j])

        self.model.encode(alpha, beta)
        self.assertGreater(len(self.model.modes), 0)

    def test_encode_spinor_signal_mismatch(self):
        """Mismatched inputs should raise error."""
        alpha = np.ones(4, dtype=complex)
        beta = np.ones(3, dtype=complex)

        with self.assertRaises(ValueError):
            self.model.encode(alpha, beta)

    def test_encode_spinor_signal_threshold(self):
        """Threshold should remove negligible modes."""
        alpha = np.zeros(8, dtype=complex)
        beta = np.zeros(8, dtype=complex)

        self.model.encode(alpha, beta, threshold=1e-6)
        self.assertEqual(len(self.model.modes), 0)

    # -- Structural Complexity --

    def test_spectral_complexity(self):
        """Verify combined component complexity."""
        # k=1 → k_eff=1 → |a|^2 + |b|^2 = 4 + 1 = 5
        self.model.modes[1] = (2+0j, 1+0j)

        # k=9 → k_eff=1 → |a|^2 + |b|^2 = 1
        self.model.modes[9] = (1+0j, 0+0j)

        expected = 5 + 1
        self.assertEqual(self.model.spectral_complexity(), expected)

    def test_spectral_complexity_empty(self):
        """Empty spectrum → zero complexity."""
        self.assertEqual(self.model.spectral_complexity(), 0.0)

    # -- Evaluation --

    def test_evaluate_shape(self):
        """Spinor field should return (N, 2)."""
        self.model.modes[1] = (1+0j, 0+1j)
        psi = self.model.evaluate()

        self.assertEqual(psi.shape, (self.N, 2))
        self.assertTrue(np.iscomplexobj(psi))

    def test_evaluate_normalization(self):
        """Evaluated spinor should be normalized."""
        self.model.modes[1] = (2+0j, 0+0j)
        psi = self.model.evaluate()

        norm = np.sum(np.abs(psi)**2)
        self.assertAlmostEqual(norm, 1.0, places=6)

    # -- Bloch Representation --

    def test_bloch_vector_shape(self):
        """Bloch vectors should be Nx3."""
        self.model.modes[1] = (1+0j, 0+1j)
        bloch = self.model.get_bloch_vectors()

        self.assertEqual(bloch.shape, (self.N, 3))

    def test_bloch_vector_bounds(self):
        """Bloch vectors should be bounded in [-1,1]."""
        self.model.modes[1] = (1+0j, 1+0j)
        bloch = self.model.get_bloch_vectors()

        self.assertTrue(np.all(bloch <= 1.0 + 1e-6))
        self.assertTrue(np.all(bloch >= -1.0 - 1e-6))

    # -- Description Length --

    def test_description_length_estimate(self):
        """Description length should increase with modes."""
        self.assertEqual(self.model.description_length_estimate(), 0.0)

        self.model.modes[1] = (2+1j, 1+0j)
        dl = self.model.description_length_estimate()
        self.assertGreater(dl, 0.0)

    # -- Typicality Weight --

    def test_typicality_weight_bounds(self):
        """Weight should decrease with complexity."""
        w0 = self.model.typicality_weight(lam=1.0)
        self.assertEqual(w0, 1.0)

        self.model.modes[2] = (10+0j, 0+0j)
        w1 = self.model.typicality_weight(lam=1.0)

        self.assertLess(w1, 1.0)


if __name__ == "__main__":
    unittest.main()
    
