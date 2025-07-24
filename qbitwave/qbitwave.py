import random
import numpy as np
from typing import Optional
from numba import njit



@njit
def bits_to_signed_float_unsigned(bits: int, length: int) -> float:
    """
    Convert an unsigned integer bit pattern into a signed float in [-1, 1).

    Args:
        bits (int): The integer value representing bits.
        length (int): Number of bits in the integer.

    Returns:
        float: Signed float value normalized to [-1, 1).
    """

    max_val = 2 ** length - 1
    return (bits / max_val) * 2 - 1


@njit
def interpret_as_wavefunction(bitarray: np.ndarray, block_size: int) -> np.ndarray:
    """
    Interpret a binary array as a wavefunction with complex amplitudes.

    Each block of bits is split into two halves representing real and imaginary parts.
    Each half-block is converted to a signed float in [-1,1). Blocks are normalized.

    Args:
        bitarray (np.ndarray): 1D array of bits (0 or 1).
        block_size (int): Number of bits per block (must be even).

    Returns:
        np.ndarray: Array of complex amplitudes normalized to unit norm.
    """

    if block_size <= 0: 
        return np.zeros(0, dtype=np.complex64)
    n = len(bitarray)
    if n % block_size != 0:
        return np.zeros(0, dtype=np.complex64)

    step = block_size
    half = step // 2
    n_blocks = len(bitarray) // step
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
    """
    Compute a normalized Shannon entropy score of the probability distribution
    derived from the given amplitudes.

    Args:
        amps (np.ndarray): Array of complex amplitudes.

    Returns:
        float: Entropy score normalized by number of amplitudes.
    """
    probs = np.abs(amps) ** 2
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy / len(amps)


class QBitwave:
    """
    Class representing a quantum-like wavefunction derived from a binary bitstring.

    The bitstring is parsed into blocks of bits, which are interpreted as complex
    amplitudes. The class can either find an optimal block size that maximizes
    the Shannon entropy of the amplitude probabilities, or use a fixed block size
    if provided.

    Attributes:
        bitstring (list[int]): The binary sequence representing the state.
        fixed_block_size (Optional[int]): If set, use this block size directly.
        amplitudes (np.ndarray): Complex amplitudes representing the wavefunction.
        selected_block_size (Optional[int]): The block size selected as optimal or fixed.
    """


    def __init__(self, bitstring: str, fixed_block_size: Optional[int] = None):
        """
        Initialize the QBitwave with a bitstring and optional fixed block size.

        Args:
            bitstring (str): Initial bitstring as a sequence of '0' and '1' characters.
            fixed_block_size (int, optional): If given, use this block size directly.
                                              Must be even and divide length of bitstring.
                                              If None, block size is found automatically.
        """
        self.bitstring = [int(b) for b in bitstring]
        self.fixed_block_size = fixed_block_size
        self.amplitudes = np.zeros(0, dtype=np.complex64)
        self.selected_block_size = None
        self._analyze_bitstring()

    def entropy(self) -> float:
        """
        Compute the Shannon entropy of the probability distribution of amplitudes.

        Returns:
            float: Shannon entropy of |amplitudes|^2.
        """
        probs = np.abs(self.amplitudes) ** 2
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    def num_states(self) -> int:
        """
        Return the number of amplitude states (length of the wavefunction).

        Returns:
            int: Number of amplitudes.
        """
        return len(self.amplitudes)

    def dimension(self) -> int:
        """
        Alias for number of states.

        Returns:
            int: Number of amplitudes.
        """
        return len(self.amplitudes)

    def norm(self) -> float:
        """
        Compute the L2 norm of the amplitudes.

        Returns:
            float: Norm value.
        """
        return float(np.sqrt(np.sum(self.amplitudes.real ** 2 + self.amplitudes.imag ** 2)))

    def get_probability_distribution(self) -> np.ndarray:
        """
        Get the probability distribution (squared magnitudes) of amplitudes.

        Returns:
            np.ndarray: Probability distribution array.
        """
        return np.abs(self.amplitudes) ** 2

    def get_phase_distribution(self) -> np.ndarray:
        """
        Get the phase angles of the amplitudes.

        Returns:
            np.ndarray: Phase angle array in radians.
        """
        return np.angle(self.amplitudes)

    def __str__(self) -> str:
        """
        Return a human-readable string representing the wavefunction as a
        superposition of basis states with significant amplitudes.

        Returns:
            str: Wavefunction string.
        """
        lines: list[str] = []
        for i, amp in enumerate(self.amplitudes):
            prob = abs(amp) ** 2
            if prob < 1e-6:
                continue
            lines.append(f"{amp:.3f} |{i:0{self._bitwidth()}b}⟩")
        return " + ".join(lines) if lines else "∅"

    def _bitwidth(self) -> int:
        """
        Compute the bit-width required to represent the indices of the amplitudes.

        Returns:
            int: Number of bits needed.
        """
        n = len(self.amplitudes)
        return max(1, int(np.ceil(np.log2(n))))

    def _interpret_as_wavefunction(self, block_size: int) -> list[complex]:
        """
        Interpret the current bitstring as a wavefunction with the given block size.

        Args:
            block_size (int): Block size to use.

        Returns:
            list[complex]: list of complex amplitudes.
        """
        if block_size is None or block_size < 2:
            return []

        bitarray = np.array(self.bitstring, dtype=np.uint8)
        amps = interpret_as_wavefunction(bitarray, block_size)
        return list(amps)  




    def _analyze_bitstring(self) -> None:
        """
        Analyze the current bitstring to find the block size that maximizes
        the entropy score of the amplitudes, or use fixed block size if set.

        Sets the amplitudes and selected_block_size accordingly.
        """

        bitarray = np.array(self.bitstring, dtype=np.uint8)
        n = len(bitarray)

        if self.fixed_block_size is not None:
            if (self.fixed_block_size > 0 and
                self.fixed_block_size % 2 == 0 and
                n % self.fixed_block_size == 0):
                amps = interpret_as_wavefunction(bitarray, self.fixed_block_size)
                if len(amps) == 1 and amps[0] == complex(-1, 0):
                    # Parsing failed
                    self.amplitudes = np.zeros(0, dtype=np.complex64)
                    self.selected_block_size = None
                    return
                self.amplitudes = amps
                self.selected_block_size = self.fixed_block_size
                return
            # fixed block size invalid
            self.amplitudes = np.zeros(0, dtype=np.complex64)
            self.selected_block_size = None
            return

        # search for best block size
        best_score = -np.inf
        best_amplitudes = np.zeros(0, dtype=np.complex64)
        best_block_size = None

        for block_size in range(2, n + 1, 2):
            if n % block_size != 0:
                continue
            amps = interpret_as_wavefunction(bitarray, block_size)
            if len(amps) == 1 and amps[0] == complex(-1, 0):
                continue
            if len(amps) == 0:
                continue
            score = score_amplitudes(amps)
            if score > best_score:
                best_score = score
                best_amplitudes = amps
                best_block_size = block_size

        if best_block_size is None:
            self.amplitudes = np.zeros(0, dtype=np.complex64)
            self.selected_block_size = None
        else:
            self.amplitudes = best_amplitudes
            self.selected_block_size = best_block_size

    def set_bitstring(self, bitstring: str) -> None:
        """
        Set a new bitstring and re-analyze it to update the wavefunction.

        Args:
            bitstring (str): New bitstring as a string of '0' and '1'.
        """
        self.bitstring = [int(b) for b in bitstring]
        self._analyze_bitstring()

    def get_amplitudes(self) -> list[complex]:
        """
        Get the current wavefunction amplitudes as a list of complex numbers.

        Returns:
            list[complex]: list of amplitudes.
        """
        return list(self.amplitudes)
    
    def get_selected_block_size(self) -> Optional[int]:
        """
        Get the block size currently selected as optimal.

        Returns:
            Optional[int]: Block size or None if not set.
        """
        return self.selected_block_size

    

    def bit_entropy(self) -> float:
        """
        Compute Shannon entropy of the bitstring itself.

        Returns:
            float: Bit-level entropy in bits.
        """
        bitarray = np.array(self.bitstring, dtype=np.uint8)
        p0 = np.count_nonzero(bitarray == 0) / len(bitarray)
        p1 = 1.0 - p0

        entropy = 0.0
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        return entropy


    def flip(self) -> None:
        """
        Flip a bit at random index in the bitstring.

        """
        index = random.randint(0, len(self.bitstring) - 1)
        self.bitstring[index] ^= 1  # flip bit
        self._analyze_bitstring()

    def mutate(self, mutation_rate: float = 0.01) -> None:
        """
        Randomly flip bits in the bitstring according to the mutation rate.

        Each bit has a chance equal to mutation_rate to be flipped.

        Args:
            mutation_rate (float, optional): Probability of flipping each bit. Defaults to 0.01.
        """
        for i in range(len(self.bitstring)):
            if random.random() < mutation_rate:
                self.bitstring[i] = 1 - self.bitstring[i]
        self._analyze_bitstring()

    
