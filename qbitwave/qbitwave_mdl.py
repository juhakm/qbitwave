"""
qbitwave_mdl.py

Finite spectral informational model (QBitwave).

Wavefunction is represented as a finite spectral set:

    {(k_i, A_i, phi_i)}

k_i : integer frequency index in Z_N
A_i : real amplitude
phi_i : phase in radians

Complexity functional:

    C_Q = sum (k_eff^2 * |A_i|^2)

Typicality weight:

    P(Psi) ∝ exp(-λ C_Q)

Phase does not contribute to structural complexity
by design (amplitude-dominant informational model).
"""

from typing import List, Tuple
import numpy as np


class QBitwaveMDL:
    """Finite spectral history encoding with genuine complex wavefunction."""

    def __init__(self, N: int):
        """
        Args:
            N: Size of discrete spatial domain (Z_N).
        """
        self.N = N
        self.modes: List[Tuple[int, float, float]] = []

    # ------------------------------------------------------------------
    # Mode Management
    # ------------------------------------------------------------------

    def add_mode(self, k: int, A: float, phi: float) -> None:
        """
        Adds spectral mode.

        Args:
            k: Integer frequency index (Z_N).
            A: Real amplitude.
            phi: Phase in radians.
        """
        k_mod = k % self.N
        self.modes.append((k_mod, float(A), float(phi)))

    def clear_modes(self) -> None:
        """Removes all spectral modes."""
        self.modes = []

    # ------------------------------------------------------------------
    # Structural Complexity (Compression Functional)
    # ------------------------------------------------------------------

    def spectral_complexity(self) -> float:
        """
        Computes structural complexity.

        C_Q = sum (k_eff^2 * |A|^2)

        where k_eff is minimal frequency distance in Z_N.
        """

        total = 0.0

        for k, A, _ in self.modes:
            k_eff = min(abs(k), self.N - abs(k))
            total += (k_eff ** 2) * (abs(A) ** 2)

        return total

    # ------------------------------------------------------------------
    # Optional Bit-Length Model (If Needed)
    # ------------------------------------------------------------------

    @staticmethod
    def bits_estimate(x: float, eps: float = 1e-12) -> float:
        """
        Rough description length estimate for real number.

        Not literal Shannon coding — just logarithmic scale cost.
        """
        return np.log2(1 + abs(x) + eps)

    def description_length_estimate(self) -> float:
        """
        Optional MDL-style description length.

        Estimates bits needed to encode (k, A, phi).
        """
        total = 0.0

        for k, A, phi in self.modes:
            total += np.log2(1 + abs(k))
            total += self.bits_estimate(A)
            total += self.bits_estimate(phi)

        return total

    # ------------------------------------------------------------------
    # Wavefunction Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> np.ndarray:
        """
        Evaluates discrete complex wavefunction ψ(x).
        """

        x_vals = np.arange(self.N)
        psi = np.zeros(self.N, dtype=complex)

        for k, A, phi in self.modes:
            phase = (2 * np.pi * k * x_vals / self.N) + phi
            psi += A * np.exp(1j * phase)

        return psi

    # ------------------------------------------------------------------
    # Born Rule Probabilities
    # ------------------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """
        Computes normalized Born probabilities:

            P(x) = |ψ(x)|^2 / sum_x |ψ(x)|^2
        """

        psi = self.evaluate()
        prob = np.abs(psi) ** 2
        total = np.sum(prob)

        if total == 0:
            return np.ones(self.N) / self.N

        return prob / total

    # ------------------------------------------------------------------
    # Typicality Weight
    # ------------------------------------------------------------------

    def typicality_weight(self, lam: float = 1.0) -> float:
        """
        Computes statistical weight:

            exp(-λ C_Q)
        """

        C_Q = self.spectral_complexity()
        return np.exp(-lam * C_Q)
