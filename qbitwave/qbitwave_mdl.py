"""
qbitwave_mdl.py

Finite spectral informational model (QBitwave).

This module implements a discrete integer spectral representation of a
wavefunction encoding a full observer history.

The wavefunction is represented as a finite set of integer modes:

    {(k_i, A_i, phi_i)}

All ontological quantities are integers.

The spectral complexity C_Q is defined as total binary description length:

    C_Q = sum bits(k_i) + bits(A_i) + bits(phi_i)

where:

    bits(n) = floor(log2(|n|)) + 1   for n != 0
    bits(0) = 1

Typicality weight:

    P(Psi) ∝ 2^{-C_Q}

Smooth dynamics emerge statistically from compression dominance.
"""

from typing import List, Tuple
import numpy as np


class QBitwaveMDL:
    """Finite integer spectral history encoding.

    The wavefunction is a static spectral object encoding a full
    configuration-space history.

    Attributes:
        N (int): Size of discrete spatial domain.
        modes (List[Tuple[int, int, int]]): List of (k, A, phi).
    """

    def __init__(self, N: int):
        """
        Initializes empty spectral representation.

        Args:
            N: Size of discrete spatial domain (Z_N).
        """
        self.N = N
        self.modes: List[Tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Mode Management
    # ------------------------------------------------------------------

    def add_mode(self, k: int, A: int, phi: int) -> None:
        """Adds integer spectral mode.

        Args:
            k: Integer frequency index (in Z_N).
            A: Integer amplitude coefficient.
            phi: Integer phase (interpreted modulo 2π scale).
        """
        k_mod = k % self.N
        self.modes.append((k_mod, int(A), int(phi)))

    def clear_modes(self) -> None:
        """Removes all spectral modes."""
        self.modes = []

    # ------------------------------------------------------------------
    # Bit-Length Function
    # ------------------------------------------------------------------

    @staticmethod
    def bits(n: int) -> int:
        """Computes binary description length of integer.

        bits(n) = floor(log2(|n|)) + 1 for n != 0
        bits(0) = 1

        Args:
            n: Integer value.

        Returns:
            int: Bit length.
        """
        if n == 0:
            return 1
        return int(np.floor(np.log2(abs(n)))) + 1

    def spectral_complexity(self) -> float:
        """
        Computes structural complexity of wavefunction.

        mode = "quadratic"   → sum k^2 A^2  (action-like)
        """

        total = 0.0

        for k, A, phi in self.modes:
            k_eff = min(abs(k), self.N - abs(k))  # minimal frequency in Z_N
            total += (k_eff ** 2) * (A ** 2)

        return total
    
    # ------------------------------------------------------------------
    # Wavefunction Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> np.ndarray:
        """Evaluates discrete wavefunction on domain Z_N.

        The evaluation algorithm is not counted in complexity.

        ψ(x) = sum A_i * exp(2π i k_i x / N + i phi_i)

        Returns:
            np.ndarray: Complex array of size N.
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
        """Computes normalized Born probabilities.

        P(x) = |ψ(x)|^2 / sum_x |ψ(x)|^2

        Returns:
            np.ndarray: Probability distribution over Z_N.
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

    def typicality_weight(self, lam : float=1.0) -> float:
        """Computes statistical weight 2^{-C_Q}.

        Returns:
            float: Typicality weight.
        """
        C_Q = self.spectral_complexity()
        return np.exp(-lam * C_Q)
