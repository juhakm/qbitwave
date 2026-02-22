import unittest
import numpy as np
from qbitwave.qbitwave_mdl import QBitwaveMDL

class TestQBitwaveMDL(unittest.TestCase):

    def setUp(self):
        """Initialize a standard model for testing."""
        self.N = 10
        self.model = QBitwaveMDL(self.N)

    def test_initialization(self):
        """Verify the domain size and empty state."""
        self.assertEqual(self.model.N, 10)
        self.assertEqual(len(self.model.modes), 0)

    def test_add_mode_logic(self):
        """Ensure modes are added and k is modulo N."""
        # Test basic addition
        self.model.add_mode(k=2, A=5, phi=0)
        self.assertEqual(self.model.modes[0], (2, 5, 0))
        
        # Test k % N (k=12 on N=10 should be k=2)
        self.model.add_mode(k=12, A=1, phi=1)
        self.assertEqual(self.model.modes[1][0], 2)


    def test_spectral_complexity(self):
        """Verify the (k^2 * A^2) complexity calculation."""
        self.model.add_mode(k=1, A=2, phi=0) # (1^2 * 2^2) = 4
        self.model.add_mode(k=9, A=1, phi=0) # k=9 on N=10 has k_eff=1. (1^2 * 1^2) = 1
        
        # Total should be 4 + 1 = 5.0
        self.assertEqual(self.model.spectral_complexity(), 5.0)

    def test_wavefunction_evaluation_shape(self):
        """Ensure the evaluated psi has correct dimensions and type."""
        self.model.add_mode(k=1, A=1, phi=0)
        psi = self.model.evaluate()
        self.assertEqual(len(psi), self.N)
        self.assertTrue(np.iscomplexobj(psi))

    def test_probabilities_normalization(self):
        """Ensure Born rule probabilities sum to 1."""
        self.model.add_mode(k=1, A=3, phi=1)
        self.model.add_mode(k=3, A=2, phi=5)
        probs = self.model.probabilities()
        
        self.assertAlmostEqual(np.sum(probs), 1.0, places=7)
        self.assertTrue(np.all(probs >= 0))

    def test_typicality_weight_bounds(self):
        """Ensure typicality weight behaves like a probability weight."""
        # Empty model (Complexity 0) -> exp(0) = 1.0
        weight_empty = self.model.typicality_weight(lam=1.0)
        self.assertEqual(weight_empty, 1.0)
        
        # Adding complexity should decrease weight
        self.model.add_mode(k=2, A=10, phi=0)
        weight_complex = self.model.typicality_weight(lam=1.0)
        self.assertLess(weight_complex, 1.0)

if __name__ == '__main__':
    unittest.main()
    
