import numpy as np
from typing import List, Optional

class QBitwave:
    def __init__(self, bitstring: str):
        self.bitstring: str = bitstring
        self.max_block_size: int = 32  # adjustable upper bound
        self.min_block_size: int = 4   # minimum size to interpret anything
        self.amplitudes: List[complex] = []
        self.selected_block_size: Optional[int] = None

        self._analyze_bitstring()

    def entropy(self) -> float:
        """Return Shannon entropy of the probability distribution |ψ|²."""
        probs: np.ndarray = np.abs(self.amplitudes) ** 2
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    def num_states(self) -> int:
        """Return the number of discrete quantum states (amplitudes)."""
        return len(self.amplitudes)

    def dimension(self) -> int:
        """Dimension of the Hilbert space."""
        return len(self.amplitudes)

    def norm(self) -> float:
        """Return the L2 norm of the wavefunction (should be ~1)."""
        return float(np.linalg.norm(self.amplitudes))

    def get_probability_distribution(self) -> np.ndarray:
        """Return the |ψ|² values as a NumPy array."""
        return np.abs(self.amplitudes) ** 2

    def get_phase_distribution(self) -> np.ndarray:
        """Return the phase angles in radians."""
        return np.angle(self.amplitudes)

    def __str__(self) -> str:
        """Pretty-print the wavefunction in Dirac-style notation."""
        lines: List[str] = []
        for i, amp in enumerate(self.amplitudes):
            prob = abs(amp) ** 2
            if prob < 1e-6:
                continue  # skip negligible
            lines.append(f"{amp:.3f} |{i:0{self._bitwidth()}b}⟩")
        return " + ".join(lines) if lines else "∅"

    def _bitwidth(self) -> int:
        """Helper: number of bits needed to label states."""
        n = len(self.amplitudes)
        return max(1, int(np.ceil(np.log2(n))))

    def _analyze_bitstring(self) -> None:
        best_score: float = -np.inf
        best_amplitudes: List[complex] = []
        best_block_size: Optional[int] = None

        for block_size in range(self.min_block_size, self.max_block_size + 1, 2):
            amps = self._interpret_as_wavefunction(block_size)
            if not amps:
                continue

            score = self._score_amplitudes(amps)

            if score > best_score:
                best_score = score
                best_amplitudes = amps
                best_block_size = block_size

        self.amplitudes = best_amplitudes
        self.selected_block_size = best_block_size

    def _interpret_as_wavefunction(self, block_size: int) -> List[complex]:
        if len(self.bitstring) < block_size:
            return []

        step = block_size
        half = step // 2
        n_blocks = len(self.bitstring) // step
        amplitudes: List[complex] = []

        for i in range(n_blocks):
            chunk = self.bitstring[i * step : (i + 1) * step]
            real_bits = chunk[:half]
            imag_bits = chunk[half:]

            re = self._bits_to_signed_float(real_bits)
            im = self._bits_to_signed_float(imag_bits)
            amplitudes.append(complex(re, im))

        norm = np.linalg.norm(amplitudes)
        if norm == 0:
            return []

        return [amp / norm for amp in amplitudes]

    def _bits_to_signed_float(self, bits: str) -> float:
        if len(bits) == 0:
            return 0.0
        max_val = 2 ** (len(bits) - 1)
        val = int(bits, 2)
        if val >= max_val:
            val -= 2 * max_val
        return val / max_val  # in range [-1, 1)

    def _score_amplitudes(self, amps: List[complex]) -> float:
        probs: np.ndarray = np.abs(amps) ** 2
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy / len(amps))  # normalize per amplitude

    def get_amplitudes(self) -> List[complex]:
        return self.amplitudes

    def get_selected_block_size(self) -> Optional[int]:
        return self.selected_block_size
