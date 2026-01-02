import unittest
import numpy as np
from qbitwave.qbitwavend import QBitwaveND

class TestQBitwaveND(unittest.TestCase):
    """
    Unit tests for the QBitwaveND class (N-dimensional dynamic wavefunction).
    """

    def setUp(self):
        # Simple 2x2 complex array for tests
        self.data = np.array([[1+0j, 0+1j],
                              [0-1j, 1+0j]], dtype=np.complex128)
        self.qnd = QBitwaveND(self.data)

    def test_initialization(self):
        """Check that amplitudes, shape, and ndim are correctly initialized."""
        q = self.qnd
        self.assertIsInstance(q.amplitudes, np.ndarray)
        self.assertEqual(q.shape, (2, 2))
        self.assertEqual(q.ndim, 2)
        self.assertIsInstance(q.fft_coeffs, np.ndarray)
        self.assertEqual(len(q.freqs), 2)

    def test_fft_coefficients_normalization(self):
        """FFT coefficients should be normalized by product of shape."""
        q = self.qnd
        expected_norm = np.fft.fftn(self.data) / np.prod(q.shape)
        np.testing.assert_allclose(q.fft_coeffs, expected_norm)

    def test_evaluate_shape_and_type(self):
        """Evaluate returns a complex scalar for given coordinates."""
        val = self.qnd.evaluate(0.5, 0.5)
        self.assertIsInstance(val, np.complex128)

    def test_evaluate_multiple_times_consistency(self):
        """Evaluate at same coordinates and t gives consistent results."""
        val1 = self.qnd.evaluate(0.1, 0.1, t=0.5)
        val2 = self.qnd.evaluate(0.1, 0.1, t=0.5)
        np.testing.assert_allclose(val1, val2)

    def test_probability_nonnegative(self):
        """Probability is non-negative and finite."""
        p = self.qnd.probability(0.0, 0.0)
        self.assertGreaterEqual(p, 0.0)
        self.assertTrue(np.isfinite(p))

    def test_time_evolve_coeffs_identity_at_t0(self):
        """At t=0, time_evolve_coeffs returns original FFT coefficients."""
        coeffs_t0 = self.qnd.time_evolve_coeffs(0)
        np.testing.assert_allclose(coeffs_t0, self.qnd.fft_coeffs)

    def test_time_evolve_coeffs_phase_modulation(self):
        """At t>0, coefficients are modulated by a phase factor."""
        t = 1.0
        coeffs_t = self.qnd.time_evolve_coeffs(t)
        self.assertEqual(coeffs_t.shape, self.qnd.fft_coeffs.shape)
        # Magnitude should remain same (unitary evolution)
        np.testing.assert_allclose(np.abs(coeffs_t), np.abs(self.qnd.fft_coeffs))

if __name__ == "__main__":
    unittest.main()
