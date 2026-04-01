"""
qbitspinor_mdl.py

Finite spectral informational model for Spinors (QBitSpinor).
The state is represented as a 2-component spectral set:
    {(k_i, A_alpha, phi_alpha), (k_i, A_beta, phi_beta)}

Structural complexity (C_Q) includes the coupling between components.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from .spectral_field_mdl import SpectralFieldMDL

class QBitSpinor(SpectralFieldMDL):
    """Spectral encoding for a 2-component (Pauli) spinor wavefunction."""

    def __init__(self, N: int):
        self.N = int(N)
        # We store modes as a dictionary mapping frequency k to (alpha_coeff, beta_coeff)
        self.modes: Dict[int, Tuple[complex, complex]] = {}

    def clear_modes(self):
        self.modes.clear()

    def encode(
        self, 
        alpha_signal: np.ndarray, 
        beta_signal: np.ndarray, 
        threshold: float = 1e-10
    ):
        """
        Encodes a 2-component spinor signal.
        alpha_signal: complex array for spin-up component
        beta_signal: complex array for spin-down component
        """
        self.clear_modes()
        
        if len(alpha_signal) != len(beta_signal):
            raise ValueError("Spinor components must have identical lengths.")

        fft_alpha = np.fft.fft(alpha_signal)
        fft_beta = np.fft.fft(beta_signal)

        for k in range(len(fft_alpha)):
            a_coeff = fft_alpha[k]
            b_coeff = fft_beta[k]
            
            # If either component has significant energy, we keep the mode
            if np.abs(a_coeff) > threshold or np.abs(b_coeff) > threshold:
                self.modes[k % self.N] = (a_coeff, b_coeff)

    def spectral_complexity(self) -> float:
        """
        Computes structural complexity for the spinor.
        C_Q = sum_k (k_eff^2 * (|A_alpha|^2 + |A_beta|^2))
        """
        total = 0.0
        for k, (a_c, b_c) in self.modes.items():
            k_eff = min(k, self.N - k)
            # Complexity is the sum of the power of both components weighted by freq
            total += (k_eff ** 2) * (np.abs(a_c)**2 + np.abs(b_c)**2)
        return total

    def get_bloch_vectors(self) -> np.ndarray:
        """
        Evaluates the expected spin direction in 3D space across the N-domain.
        Returns: [N, 3] array of (Ex, Ey, Ez)
        """
        psi = self.evaluate() # returns [N, 2] complex array
        alpha = psi[:, 0]
        beta = psi[:, 1]
        
        # Standard Pauli expectation values
        ex = 2 * np.real(np.conj(alpha) * beta)
        ey = 2 * np.imag(np.conj(alpha) * beta)
        ez = np.abs(alpha)**2 - np.abs(beta)**2
        
        return np.column_stack((ex, ey, ez))

    def evaluate(self) -> np.ndarray:
        """Evaluates the spinor wavefunction [N, 2] in Z_N."""
        x_vals = np.arange(self.N)
        psi = np.zeros((self.N, 2), dtype=complex)

        for k, (a_c, b_c) in self.modes.items():
            phase = 2 * np.pi * k * x_vals / self.N
            exp_phase = np.exp(1j * phase)
            psi[:, 0] += a_c * exp_phase
            psi[:, 1] += b_c * exp_phase

        # Normalize total probability to 1
        norm = np.sqrt(np.sum(np.abs(psi)**2))
        if norm > 0:
            psi /= norm
            
        return psi

    def description_length_estimate(self) -> float:
        """MDL bit-cost estimate for the spinor spectrum."""
        total = 0.0
        for k, (a_c, b_c) in self.modes.items():
            total += np.log2(1.0 + k)
            # Cost of alpha
            total += np.log2(1.0 + np.abs(a_c)) + np.log2(1.0 + np.abs(np.angle(a_c)))
            # Cost of beta
            total += np.log2(1.0 + np.abs(b_c)) + np.log2(1.0 + np.abs(np.angle(b_c)))
        return total

    def typicality_weight(self, lam: float = 1.0) -> float:
        return np.exp(-lam * self.spectral_complexity())

