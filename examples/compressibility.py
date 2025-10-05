"""
Wavefunction-Centric QBitwave Demo (Sine vs Noise)
==================================================

Purpose:
--------
This demo provides a concrete illustration of the principle that quantum-like
probability amplitudes can emerge purely from informational constraints.
Specifically, it tests the hypothesis that **compressibility drives
emergent amplitude distributions**: configurations that are more easily
compressed dominate the set of possible outcomes, giving rise to the
observable structure of wavefunction amplitudes.

Fundamental idea:
-----------------
- A bitstring represents the discrete, underlying information content
  of a system (Kolmogorov-inspired perspective).
- The QBitwave class interprets a bitstring as a wavefunction:
    - Amplitudes encode the magnitude of probability.
    - Phases encode relative interference.
- Smooth, structured bitstrings (e.g., sine wave) contain correlations
  that make them highly compressible; as a result, they produce
  strong, dominant amplitudes in the emergent wavefunction.
- Random, unstructured bitstrings are incompressible; they generate
  diffuse, weak amplitudes corresponding to low predictability.

Conceptual significance:
------------------------
- This demo visually and quantitatively contrasts compressible vs.
  incompressible bitstrings.
- It confirms the principle that **emergent quantum probability amplitudes**
  can be viewed as a natural consequence of algorithmic information theory:
  the "probability" of observing a configuration is proportional to
  its compressibility (shortest effective program length).
- The sine vs. noise comparison highlights how structured patterns
  dominate the ensemble of possible configurations, while random noise
  is evenly distributed and thus appears "flat" in amplitude.

Demo details:
-------------
- Left panel: sine-wave bitstring → smooth, highly compressible amplitude
  distribution, reflecting coherent structure.
- Right panel: random bitstring → low compressibility, diffuse amplitude
  distribution, reflecting lack of structure.
- Compressibility is quantified via Fourier-domain analysis of the
  emergent wavefunction.

Relation to broader framework:
-------------------------------
This demo is one instance in a suite of programs exploring the same
fundamental principle: **quantum behavior emerges from selection over
algorithmically simple (compressible) configurations**.
In this sense, compressibility = probability amplitude = predictability.
The underlying physics (interference, probability distribution) is not
assumed, but emerges naturally from informational constraints.

Author:
-------
(c) 2019–2025 Juha Meskanen
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
import random
from qbitwave import QBitwave

# -----------------------------
# Utilities
# -----------------------------
def generate_sine_bitstring(num_samples: int, bits_per_amplitude: int = 8) -> List[int]:
    """
    Generate a sine wave bitstring with given number of samples and bits per amplitude.
    """
    t = np.linspace(0, 1, num_samples, endpoint=False)
    sine_wave = 0.5 + 0.5 * np.sin(2 * np.pi * t)
    bitstring: List[int] = []
    for amp in sine_wave:
        int_val = int(round(amp * (2**bits_per_amplitude - 1)))
        bits = [(int_val >> i) & 1 for i in reversed(range(bits_per_amplitude))]
        bitstring.extend(bits)
    return bitstring

def generate_noise_bitstring(length: int) -> List[int]:
    """
    Generate a random bitstring of given length.
    """
    return [random.randint(0, 1) for _ in range(length)]


def plot_side_by_side(qbw1: QBitwave, title1: str, qbw2: QBitwave, title2: str) -> None:
    """
    Display two wavefunction heatmaps side by side, with a separate rightmost colorbar.
    """
    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cbar_ax = fig.add_subplot(gs[0, 2])

    for ax, qbw, title in zip(axes, [qbw1, qbw2], [title1, title2]):
        amps = np.array(qbw.get_amplitudes(), dtype=np.complex128)

        if len(amps) > 4:
            amps = np.convolve(amps.real, np.ones(4)/4, mode='same') + \
                   1j * np.convolve(amps.imag, np.ones(4)/4, mode='same')

        n = len(amps)
        dim = int(np.ceil(np.sqrt(n)))
        padded = np.zeros(dim * dim, dtype=np.float64)
        padded[:n] = np.abs(amps)
        image = padded.reshape((dim, dim))

        im = ax.imshow(image, cmap="viridis", interpolation="bilinear")
        ax.set_title(f"{title}\nCompressibility: {qbw.compressibility():.3f}")
        ax.axis("off")

    # Attach colorbar to the separate column
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="Amplitude")
    plt.show()

# -----------------------------
# Main demo
# -----------------------------
def main(num_samples: int = 512, bits_per_amplitude: int = 8):
    # Generate bitstrings
    sine_bits = generate_sine_bitstring(num_samples, bits_per_amplitude)
    noise_bits = generate_noise_bitstring(num_samples * bits_per_amplitude)

    # Construct QBitwave objects
    qb_sine = QBitwave(bitstring=sine_bits, fixed_basis_size=bits_per_amplitude)
    qb_noise = QBitwave(bitstring=noise_bits, fixed_basis_size=bits_per_amplitude)

    # Plot side by side
    plot_side_by_side(qb_sine, "Sine Wave Bitstring", qb_noise, "Random Noise Bitstring")

    # Print compressibility using class method
    print(f"Sine wave compressibility: {qb_sine.compressibility():.3f}")
    print(f"Random noise compressibility: {qb_noise.compressibility():.3f}")

# -----------------------------
# Startup args
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wavefunction-Centric QBitwave Demo")
    parser.add_argument("--samples", type=int, default=512, help="Number of sine samples")
    parser.add_argument("--bits", type=int, default=8, help="Bits per amplitude block")
    args = parser.parse_args()

    main(num_samples=args.samples, bits_per_amplitude=args.bits)
