import unittest
import numpy as np
from qbitwave.qbitwave_mdl import QBitwaveMDL


class TestQBitwaveMDL(unittest.TestCase):

    def setUp(self):
        """Initialize a standard model for testing."""
        self.N = 10
        self.model = QBitwaveMDL(self.N)

    # -- Initialization and Mode Management --

    def test_initialization(self):
        """Verify the domain size and empty state."""
        self.assertEqual(self.model.N, 10)
        self.assertEqual(len(self.model.modes), 0)

    def test_add_mode_logic(self):
        """Ensure modes are added and k is reduced modulo N."""
        self.model.add_mode(k=2, A=5, phi=0)
        self.assertEqual(self.model.modes[0], (2, 5.0, 0.0))

        self.model.add_mode(k=12, A=1, phi=1)
        self.assertEqual(self.model.modes[1][0], 2)

    def test_clear_modes(self):
        """Ensure modes are completely cleared."""
        self.model.add_mode(k=1, A=1, phi=0)
        self.model.clear_modes()
        self.assertEqual(len(self.model.modes), 0)

    # -- FFT Encoding -- 

    def test_encode_complex_signal_basic(self):
        """FFT encoding should populate modes for a valid signal."""
        z = np.array([1 + 1j, -1 - 1j, 1 + 1j, -1 - 1j])
        self.model.encode_complex_signal(z)
        self.assertGreater(len(self.model.modes), 0)

    def test_encode_complex_signal_short_input(self):
        """Signals shorter than length 2 should produce no modes."""
        z = np.array([1 + 1j])
        self.model.encode_complex_signal(z)
        self.assertEqual(len(self.model.modes), 0)

    def test_encode_complex_signal_threshold(self):
        """Amplitude threshold should suppress near-zero modes."""
        z = np.zeros(8, dtype=complex)
        self.model.encode_complex_signal(z, amplitude_threshold=1e-6)
        self.assertEqual(len(self.model.modes), 0)

    # -- Structural Complexity --

    def test_spectral_complexity(self):
        """Verify the k_eff^2 * A^2 complexity calculation."""
        self.model.add_mode(k=1, A=2, phi=0)  # 1^2 * 4 = 4
        self.model.add_mode(k=9, A=1, phi=0)  # k_eff=1 → 1
        self.assertEqual(self.model.spectral_complexity(), 5.0)

    def test_spectral_complexity_empty(self):
        """Empty spectrum should have zero complexity."""
        self.assertEqual(self.model.spectral_complexity(), 0.0)

    # -- Spectrum Diagnostics --

    def test_spectrum_output(self):
        """Spectrum should return k_eff, weighted power, raw power."""
        self.model.add_mode(k=2, A=3, phi=0)
        k_eff, weighted, raw = self.model.spectrum()

        self.assertEqual(len(k_eff), 1)
        self.assertEqual(raw[0], 9.0)
        self.assertEqual(weighted[0], (2 ** 2) * 9.0)

    def test_spectrum_empty(self):
        """Empty model should return (None, None, None)."""
        self.assertEqual(self.model.spectrum(), (None, None, None))

    # -- Wavefunction Evaluation --

    def test_wavefunction_evaluation_shape(self):
        """Ensure evaluated ψ has correct shape and type."""
        self.model.add_mode(k=1, A=1, phi=0)
        psi = self.model.evaluate()
        self.assertEqual(len(psi), self.N)
        self.assertTrue(np.iscomplexobj(psi))

    # -- Born Probabilities --

    def test_probabilities_normalization(self):
        """Born probabilities should sum to 1."""
        self.model.add_mode(k=1, A=3, phi=1)
        self.model.add_mode(k=3, A=2, phi=5)
        probs = self.model.probabilities()

        self.assertAlmostEqual(np.sum(probs), 1.0, places=7)
        self.assertTrue(np.all(probs >= 0))

    def test_probabilities_empty(self):
        """Empty wavefunction should give uniform distribution."""
        probs = self.model.probabilities()
        self.assertTrue(np.allclose(probs, np.ones(self.N) / self.N))

    # -- Bit-Length / MDL Utilities --

    def test_bits_estimate_static(self):
        """bits_estimate should be callable statically and monotonic."""
        b1 = QBitwaveMDL.bits_estimate(1.0)
        b2 = QBitwaveMDL.bits_estimate(10.0)
        self.assertGreater(b2, b1)

    def test_description_length_estimate(self):
        """Description length should increase with more modes."""
        self.assertEqual(self.model.description_length_estimate(), 0.0)

        self.model.add_mode(k=1, A=2, phi=3)
        dl = self.model.description_length_estimate()
        self.assertGreater(dl, 0.0)

    # -- Typicality Weight --

    def test_typicality_weight_bounds(self):
        """Typicality weight should decrease with complexity."""
        w0 = self.model.typicality_weight(lam=1.0)
        self.assertEqual(w0, 1.0)

        self.model.add_mode(k=2, A=10, phi=0)
        w1 = self.model.typicality_weight(lam=1.0)
        self.assertLess(w1, 1.0)


if __name__ == "__main__":
    unittest.main()

