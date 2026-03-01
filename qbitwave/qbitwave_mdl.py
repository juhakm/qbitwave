"""
qbitwave_mdl.py

Finite spectral informational model (QBitwave).

Wavefunctions are represented as a finite spectral set:

    {(k_i, A_i, phi_i)}

where:
    k_i   : integer frequency index in Z_N
    A_i   : real amplitude
    phi_i : phase in radians

Structural complexity functional:

    C_Q = sum (k_eff^2 * |A_i|^2)

Typicality weight:

    P(Psi) ∝ exp(-λ C_Q)

Phase does not contribute to structural complexity
by design (amplitude-dominant informational model).
"""

from typing import List, Tuple, Optional
import numpy as np


class QBitwaveMDL:
    """Finite spectral history encoding with a genuine complex wavefunction.

    This class is the *sole owner* of:
    - FFT-based encoding
    - Spectral complexity (MDL cost)
    - k-weighted power spectra

    No external class should perform FFTs directly.
    """

    def __init__(self, N: int):
        """
        Args:
            N: Maximum number of spectral modes (Z_N domain).
        """
        self.N = int(N)
        self.modes: List[Tuple[int, float, float]] = []


    def add_mode(self, k: int, A: float, phi: float) -> None:
        """Adds a spectral mode.

        Args:
            k: Integer frequency index.
            A: Real amplitude.
            phi: Phase in radians.
        """
        k_mod = k % self.N
        self.modes.append((k_mod, float(A), float(phi)))

    def clear_modes(self) -> None:
        """Removes all spectral modes."""
        self.modes.clear()


    def encode_complex_signal(
        self,
        z: np.ndarray,
        amplitude_threshold: float = 1e-10
    ) -> None:
        """Encodes a complex-valued signal into spectral modes via FFT.

        This is the *canonical* entry point for trajectory encoding.

        Args:
            z: Complex signal array (e.g., x + i y trajectory).
            amplitude_threshold: Minimum amplitude to retain a mode.
        """
        self.clear_modes()

        if z is None or len(z) < 2:
            return

        fft_vals = np.fft.fft(z)
        N_fft = len(fft_vals)

        for k, coeff in enumerate(fft_vals):
            A = np.abs(coeff)
            if A < amplitude_threshold:
                continue

            phi = np.angle(coeff)
            self.add_mode(k % self.N, A, phi)


    def spectral_complexity(self) -> float:
        """Computes structural (MDL) complexity.

        Definition:
            C_Q = sum_k (k_eff^2 * |A_k|^2)

        where:
            k_eff = min(k, N - k)

        Returns:
            Scalar complexity cost.
        """
        total = 0.0

        for k, A, _ in self.modes:
            k_eff = min(k, self.N - k)
            total += (k_eff ** 2) * (A ** 2)

        return total


    def spectrum(self):
        """Returns the weighted spectral power distribution.

        Returns:
            k_eff (np.ndarray): Effective frequency indices.
            weighted_power (np.ndarray): k_eff^2 * |A_k|^2
            raw_power (np.ndarray): |A_k|^2
        """
        if not self.modes:
            return None, None, None

        ks = np.array([k for k, _, _ in self.modes], dtype=int)
        amps = np.array([A for _, A, _ in self.modes], dtype=float)

        k_eff = np.minimum(ks, self.N - ks)
        raw_power = amps ** 2
        weighted = (k_eff ** 2) * raw_power

        return k_eff, weighted, raw_power


    @staticmethod
    def bits_estimate(x: float, eps: float = 1e-12) -> float:
        """Estimates description length of a real number.

        This is a logarithmic proxy for MDL-style encoding cost.

        Args:
            x: Value to encode.
            eps: Numerical stabilizer.

        Returns:
            Approximate bit cost.
        """
        return np.log2(1.0 + abs(x) + eps)

    def description_length_estimate(self) -> float:
        """Estimates total description length of the spectrum.

        Returns:
            Approximate number of bits to encode all modes.
        """
        total = 0.0

        for k, A, phi in self.modes:
            total += np.log2(1.0 + abs(k))
            total += self.bits_estimate(A)
            total += self.bits_estimate(phi)

        return total

    
    def evaluate(self) -> np.ndarray:
        """Evaluates the discrete complex wavefunction ψ(x).

        Returns:
            Complex-valued array of length N.
        """
        x_vals = np.arange(self.N)
        psi = np.zeros(self.N, dtype=complex)

        for k, A, phi in self.modes:
            phase = (2 * np.pi * k * x_vals / self.N) + phi
            psi += A * np.exp(1j * phase)

        return psi


    def probabilities(self) -> np.ndarray:
        """Computes normalized Born-rule probabilities.

        Returns:
            Probability distribution over x ∈ Z_N.
        """
        psi = self.evaluate()
        prob = np.abs(psi) ** 2
        s = prob.sum()

        if s <= 0:
            return np.ones(self.N) / self.N

        return prob / s


    def typicality_weight(self, lam: float = 1.0) -> float:
        """Computes typicality weight exp(-λ C_Q).

        Args:
            lam: Inverse temperature / compression strength.

        Returns:
            Statistical weight.
        """
        return np.exp(-lam * self.spectral_complexity())
