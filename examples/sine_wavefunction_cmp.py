import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave


def amplitude_to_bits(amplitude: float, block_size: int) -> list[int]:
    max_int = 2**block_size - 1
    int_val = int(round(amplitude * max_int))
    bits = [(int_val >> i) & 1 for i in reversed(range(block_size))]
    return bits

def bits_to_amplitude(bits: list[int]) -> float:
    block_size = len(bits)
    max_int = 2**block_size - 1
    int_val = sum(b << (block_size - 1 - i) for i, b in enumerate(bits))
    return int_val / max_int

def generate_sine_bitstring(num_samples: int, block_size: int) -> tuple[list[int], np.ndarray]:
    t = np.linspace(0, 1, num_samples, endpoint=False)
    sine_wave = 0.5 + 0.5 * np.sin(2 * np.pi * t)
    bitstring = []
    for amp in sine_wave:
        bits = amplitude_to_bits(amp, block_size)
        bitstring.extend(bits)
    return bitstring, sine_wave

def decode_bitstring_manual(bitstring: list[int], block_size: int) -> np.ndarray:
    assert len(bitstring) % block_size == 0
    decoded = []
    for i in range(0, len(bitstring), block_size):
        block = bitstring[i:i + block_size]
        amp = bits_to_amplitude(block)
        decoded.append(amp)
    return np.array(decoded)


def run_test_matrix():
    block_sizes = [4, 32]
    sample_counts = [16, 128]

    fig, axes = plt.subplots(len(block_sizes), len(sample_counts), figsize=(16, 12))
    fig.suptitle("QBitwave vs Manual Decode vs Original Sine", fontsize=16)

    for i, block_size in enumerate(block_sizes):
        for j, num_samples in enumerate(sample_counts):
            bitstring, original_sine = generate_sine_bitstring(num_samples, block_size)
            manual_decode = decode_bitstring_manual(bitstring, block_size)
            qbit = QBitwave(bitstring, block_size)
            qbit_decode = np.real(qbit.amplitudes[:num_samples])

            x = np.linspace(0, 1, num_samples, endpoint=False)
            ax = axes[i, j]
            ax.plot(x, original_sine, label="Original", lw=1.5)
            ax.plot(x, manual_decode, label="Manual", lw=1.0, linestyle="--")
            ax.plot(x, qbit_decode, label="QBitwave", lw=1.0, linestyle=":")
            ax.set_title(f"Block {block_size}, Samples {num_samples}")
            ax.set_xticks([])
            ax.set_yticks([])

            if i == len(block_sizes) - 1:
                ax.set_xlabel("Phase")
            if j == 0:
                ax.set_ylabel("Amplitude")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    plt.savefig("qbitwave.png", dpi=300)
    plt.show()
    print("Done")


if __name__ == "__main__":
    run_test_matrix()
