
import matplotlib.pyplot as plt
import numpy as np
from qbitwave import QBitwave  # or just use your class directly


def amplitude_to_bits(amplitude: float, block_size: int) -> list[int]:
    max_int = 2**block_size - 1
    int_val = int(round(amplitude * max_int))
    bits = [(int_val >> i) & 1 for i in reversed(range(block_size))]
    return bits

def generate_sine_bitstring(num_samples: int, block_size: int) -> np.ndarray:
    t = np.linspace(0, 1, num_samples, endpoint=False)
    sine_wave = 0.5 + 0.5 * np.sin(2 * np.pi * t)
    bitstring = []
    for amp in sine_wave:
        bits = amplitude_to_bits(amp, block_size)
        bitstring.extend(bits)
    return bitstring


def plot_heatmap(qbw: QBitwave, title="Wavefunction Heatmap"):
    amps = qbw.get_amplitudes()
    probs = np.abs(np.array(amps)) ** 2

    n = len(probs)
    dim = int(np.ceil(np.sqrt(n)))
    padded_probs = np.zeros(dim * dim)
    padded_probs[:n] = probs
    image = padded_probs.reshape((dim, dim))

    plt.imshow(image, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="Probability")
    plt.axis("off")
    plt.show()


bitstring = generate_sine_bitstring(4096, 32)
qb = QBitwave(bitstring) 
plot_heatmap(qb)

