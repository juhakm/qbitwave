"""
QBitwave: Wavefunction as an Emergent Information-Theoretic Object
==================================================================

This module defines the QBitwave class, which treats the wavefunction
as an emergent, information-theoretic object. The bitstring is a
discretized representation of the wavefunction, and the wavefunction
can be recovered as the minimal, normalized complex amplitudes. 

From an algorithmic information perspective, the wavefunction
represents the Kolmogorov-minimal program producing the bitstring.

Fundamental principle:
----------------------
- Compression = quantum probability amplitude = predictability.
- Smooth, regular data compresses well and corresponds to high amplitude
  in fewer Fourier components (low entropy, high predictability).
- Random or noisy data is incompressible, yielding low amplitude concentration.
- Wavefunction = minimal program reproducing the bitstring (Kolmogorov complexity view).

Features:
---------
- Forward mapping: wavefunction → bitstring
- Reverse mapping: bitstring → minimal wavefunction
- Block-size selection via entropy maximization
- Shannon entropy computation
- Fourier-based compressibility measure reflecting structure

Author:
-------
(c) 2019-2025 Juha Meskanen
"""

from typing import List, Optional
import numpy as np
import warnings
import random
from numba import njit


@njit
def bits_to_signed_float_unsigned(bits: int, length: int) -> float:
    """Convert unsigned integer bits to signed float in [-1, 1)."""
    max_val = 2 ** length - 1
    return (bits / max_val) * 2 - 1


@njit
def interpret_as_wavefunction(bitarray: np.ndarray, basis_size: int) -> np.ndarray:
    """Interpret a binary array as a normalized complex wavefunction.

    Args:
        bitarray (np.ndarray): 1D array of bits (0 or 1).
        basis_size (int): Number of bits per amplitude block (must be even).

    Returns:
        np.ndarray: Normalized complex amplitudes of the wavefunction.
    """
    if basis_size <= 0:
        return np.zeros(0, dtype=np.complex64)
    n = len(bitarray)
    if n % basis_size != 0:
        return np.zeros(0, dtype=np.complex64)

    step = basis_size
    half = step // 2
    n_blocks = n // step
    amplitudes = np.empty(n_blocks, dtype=np.complex64)

    for i in range(n_blocks):
        start = i * step
        real_val = 0
        imag_val = 0
        for j in range(half):
            real_val = (real_val << 1) | bitarray[start + j]
            imag_val = (imag_val << 1) | bitarray[start + half + j]

        re = bits_to_signed_float_unsigned(real_val, half)
        im = bits_to_signed_float_unsigned(imag_val, half)
        amplitudes[i] = re + 1j * im

    norm = np.sqrt(np.sum(amplitudes.real ** 2 + amplitudes.imag ** 2))
    if norm == 0:
        return np.zeros(0, dtype=np.complex64)
    return amplitudes / norm


@njit
def score_amplitudes(amps: np.ndarray) -> float:
    """Compute Shannon entropy of normalized amplitude probabilities.

    Args:
        amps (np.ndarray): Array of complex amplitudes.

    Returns:
        float: Normalized entropy score in bits per amplitude.
    """
    probs = np.abs(amps) ** 2
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy / len(amps)


class QBitwave:
    """Wavefunction-centric information-theoretic representation.

    Attributes:
        bitstring (List[int]): Binary representation of information.
        amplitudes (np.ndarray): Complex amplitudes of the wavefunction.
        phases (np.ndarray): Phase angles of amplitudes.
        freqs (Optional[np.ndarray]): Optional frequency grid for forward mapping.
        fixed_basis_size (Optional[int]): Fixed block size if provided.
        selected_basis_size (Optional[int]): Block size chosen during analysis.
    """

    def __init__(self,
                 bitstring: Optional[List[int]] = None,
                 amplitudes: Optional[np.ndarray] = None,
                 phases: Optional[np.ndarray] = None,
                 freqs: Optional[np.ndarray] = None,
                 fixed_basis_size: Optional[int] = None):
        """Initialize QBitwave from either a bitstring or wavefunction.

        Args:
            bitstring (List[int], optional): Initial binary sequence.
            amplitudes (np.ndarray, optional): Complex amplitudes for forward mapping.
            phases (np.ndarray, optional): Phase angles for forward mapping.
            freqs (np.ndarray, optional): Optional frequency grid.
            fixed_basis_size (int, optional): If provided, use fixed block size.
        
        Raises:
            ValueError: If neither bitstring nor (amplitudes + phases) are provided.
        """
        self.bitstring = bitstring
        self.fixed_basis_size = fixed_basis_size
        self.selected_basis_size = None
        self.amplitudes = np.zeros(0, dtype=np.complex64)
        self.phases = np.zeros(0, dtype=np.float32)
        self.freqs = freqs

        if amplitudes is not None and phases is not None:
            self.amplitudes = amplitudes / np.sqrt(np.sum(np.abs(amplitudes)**2))
            self.phases = phases
            self.bitstring = self.to_bitstring()
            self.selected_basis_size = fixed_basis_size
        elif bitstring is not None:
            self.bitstring = [int(b) for b in bitstring]
            self._analyze_bitstring()
            self.phases = np.angle(self.amplitudes)
        else:
            raise ValueError("Must provide either bitstring or (amplitudes + phases)")

    # -----------------------------------------------------------------------
    # Forward mapping
    # -----------------------------------------------------------------------
    def to_bitstring(self, bits_per_amp: int = 8, bits_per_phase: int = 8) -> List[int]:
        """Convert current wavefunction to a bitstring representation.

        Args:
            bits_per_amp (int): Bits to encode amplitude.
            bits_per_phase (int): Bits to encode phase.

        Returns:
            List[int]: Flattened bitstring encoding amplitudes and phases.
        """
        bitstring = []
        amps = np.abs(self.amplitudes)
        max_amp = np.max(amps) if len(amps) > 0 else 1.0
        amps_norm = amps / max_amp if max_amp > 0 else amps
        for a, phi in zip(amps_norm, self.phases):
            # amplitude bits
            amp_bits = [(int(round(a * (2**bits_per_amp - 1))) >> i) & 1
                        for i in reversed(range(bits_per_amp))]
            # phase bits
            phi_bits = [(int(round(phi / (2*np.pi) * (2**bits_per_phase - 1))) >> i) & 1
                        for i in reversed(range(bits_per_phase))]
            bitstring.extend(amp_bits + phi_bits)
        return bitstring

    # -----------------------------------------------------------------------
    # Reverse mapping
    # -----------------------------------------------------------------------

    def _analyze_bitstring(self) -> None:
        """Analyze bitstring as ordered real-valued amplitudes.

        This version preserves spatial structure: consecutive bits represent
        consecutive sample amplitudes of the underlying wave. It converts each
        fixed-size block of bits into a signed float in [-1,1), then normalizes
        and treats the result as the complex amplitude array.
        """
        if self.bitstring is None or len(self.bitstring) == 0:
            self.amplitudes = np.zeros(0, dtype=np.complex64)
            self.selected_basis_size = None
            return

        if self.fixed_basis_size is not None:
            basis_size = self.fixed_basis_size
        else:
            # choose even block size automatically if not provided
            n = len(self.bitstring)
            basis_size = 8 if n % 8 == 0 else 16

        bits = np.array(self.bitstring, dtype=np.uint8)
        n = len(bits)
        n_blocks = n // basis_size

        # Convert each block of bits into a real amplitude in [-1,1)
        amps_real = np.empty(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            start = i * basis_size
            block = bits[start:start + basis_size]
            val = 0
            for b in block:
                val = (val << 1) | int(b)
            amps_real[i] = bits_to_signed_float_unsigned(val, basis_size)

        # Normalize and store as complex amplitudes
        norm = np.linalg.norm(amps_real)
        if norm == 0:
            amps_real = np.zeros_like(amps_real)
        else:
            amps_real /= norm

        self.amplitudes = amps_real.astype(np.complex64)
        self.selected_basis_size = basis_size


    def _analyze_bitstring_1(self) -> None:
        """Analyze bitstring to find optimal block size and reconstruct amplitudes."""
        if self.fixed_basis_size is not None:
            basis_sizes = [self.fixed_basis_size]
        else:
            n = len(self.bitstring)
            basis_sizes = [i for i in range(2, n+1, 2) if n % i == 0]

        best_score = -np.inf
        best_amps = np.zeros(0, dtype=np.complex64)
        best_basis = None

        bitarray = np.array(self.bitstring, dtype=np.uint8)
        for b in basis_sizes:
            amps = interpret_as_wavefunction(bitarray, b)
            if len(amps) == 0:
                continue
            score = score_amplitudes(amps)
            if score > best_score:
                best_score = score
                best_amps = amps
                best_basis = b

        self.amplitudes = best_amps
        self.selected_basis_size = best_basis

    # -----------------------------------------------------------------------
    # Core functionality
    # -----------------------------------------------------------------------
    def entropy(self) -> float:
        """Compute Shannon entropy of the amplitude probabilities.

        Returns:
            float: Shannon entropy of |amplitudes|^2.
        """
        probs = np.abs(self.amplitudes) ** 2
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    def get_amplitudes(self) -> np.ndarray:
        """Return complex amplitudes of the wavefunction.

        Returns:
            np.ndarray: Complex amplitude array.
        """
        return self.amplitudes

    def compressibility(self, threshold: float = 0.01) -> float:
        """Fourier-based compressibility measure of the wavefunction.

        Fewer significant Fourier components → higher compressibility → more
        predictable, smooth information.

        Args:
            threshold (float): Relative amplitude threshold to count significant
                               Fourier components.

        Returns:
            float: Compressibility in [0, 1], higher = more compressible.
        """
        amps = np.abs(self.amplitudes)
        if len(amps) == 0:
            return 0.0

        fft_coeffs = np.fft.fft(amps)
        max_amp = np.max(np.abs(fft_coeffs))
        sig = np.sum(np.abs(fft_coeffs) / max_amp > threshold)
        return 1.0 - sig / len(fft_coeffs)

    def set_bitstring(self, bitstring: List[int]) -> None:
        """Update bitstring and recompute wavefunction representation.

        Args:
            bitstring (List[int]): New bitstring to analyze.
        """
        self.bitstring = [int(b) for b in bitstring]
        self._analyze_bitstring()
        self.phases = np.angle(self.amplitudes)

    def flip(self, n_flips: int = 1) -> None:
        """
        Randomly flip bits in the internal bitstring.

        This method introduces small perturbations in the informational state,
        allowing exploration of nearby configurations in bitspace. It does not
        represent physical evolution but *informational mutation* — a generic,
        structure-agnostic operation.

        Args:
            n_flips (int): number of bits to randomly toggle (default = 1).

        Notes:
            After mutation, the wavefunction is automatically reanalyzed to
            reflect the new bit configuration.
        """
        if not self.bitstring or len(self.bitstring) == 0:
            return

        n = len(self.bitstring)
        for _ in range(n_flips):
            idx = np.random.randint(0, n)
            self.bitstring[idx] ^= 1  # toggle bit

        # Recompute amplitudes and phases
        self._analyze_bitstring()
        self.phases = np.angle(self.amplitudes)

    def bit_entropy(self) -> float:
        """
        Shannon entropy of the raw bitstring (syntactic entropy).

        Returns:
            float: bit-level entropy in [0, 1], normalized to 1 bit per symbol.

        Interpretation:
            - High (~1.0): random, incompressible bitstring
            - Low (~0.0): structured, compressible bitstring
        """
        if not self.bitstring or len(self.bitstring) == 0:
            return 0.0

        bits = np.array(self.bitstring, dtype=np.uint8)
        p1 = np.mean(bits)
        p0 = 1.0 - p1
        probs = np.array([p0, p1])
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log2(probs))
        return float(H)


    def coherence(self) -> float:
        """
        Relative entropy (Kullback–Leibler divergence) between bit-level and 
        wavefunction-level probability distributions.

        Concept:
        ---------
        The bitstring encodes microscopic information; the wavefunction encodes
        its emergent, macroscopic amplitude structure. If the wavefunction
        faithfully represents the underlying information, their probability
        distributions should align.

        This function computes:
            D_KL(P_bits || P_ψ) = Σ P_bits * log2(P_bits / P_ψ)

        where:
            P_bits  = [p0, p1]  — probabilities of bits (0 and 1)
            P_ψ     = normalized amplitude-based probabilities

        Interpretation:
        ----------------
        - Low coherence divergence (≈ 0): 
            The wavefunction closely matches the bit-level structure.
            High informational faithfulness → strong emergence.

        - High coherence divergence: 
            The wavefunction diverges from the bit substrate; 
            emergent structure has lost contact with underlying information.

        This serves as a measure of *informational transparency* 
        between micro (bit) and macro (wavefunction) levels — 
        essentially how much of the raw data pattern is preserved 
        in the emergent quantum description.

        Returns:
            float: Kullback–Leibler divergence in bits (≥ 0)
        """
        # Compute bit probabilities
        if not self.bitstring or len(self.bitstring) == 0:
            return 0.0

        bits = np.array(self.bitstring, dtype=np.uint8)
        p1 = np.mean(bits)
        p0 = 1.0 - p1
        P_bits = np.array([p0, p1])
        P_bits = P_bits[P_bits > 0]  # avoid log(0)

        # Compute amplitude-based probabilities
        if self.amplitudes is None or len(self.amplitudes) == 0:
            return 0.0

        P_psi = np.abs(self.amplitudes) ** 2
        P_psi /= np.sum(P_psi)
        P_psi = P_psi[P_psi > 0]

        # Align lengths if necessary (truncate to smallest)
        n = min(len(P_bits), len(P_psi))
        P_bits = P_bits[:n]
        P_psi = P_psi[:n]

        # Compute KL divergence (bits)
        Dkl = np.sum(P_bits * np.log2(P_bits / P_psi))
        return float(max(0.0, Dkl))

    def mutate(self, level=0.01):
        """
        Apply small complex noise to amplitudes, simulating informational drift.
        This increases entropy while preserving normalization.
        """
        noise = (np.random.randn(*self.amplitudes.shape) +
                1j * np.random.randn(*self.amplitudes.shape))
        self.amplitudes += level * noise
        self.amplitudes /= np.linalg.norm(self.amplitudes)
        # Recompute Fourier coefficients for future evaluations
        self.fft_coeffs = np.fft.fftn(self.amplitudes) / np.prod(self.amplitudes.shape)

