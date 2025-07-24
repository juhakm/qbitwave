import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave



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


def main():
    steps : int = 500
    num_samples : int = 128
    block_size : int = 64

    # start out something smooth, highly compressible bitpattern
    bitstring = generate_sine_bitstring(num_samples, block_size)
    qbit = QBitwave(bitstring)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_title("Wavefunction Emergence from Bitstring")
    ax.set_xlabel("Phase (arbitrary)")
    ax.set_ylabel("Amplitude")

    def update_plot():
        amps = qbit.amplitudes
        if amps.size > 0:
            x = np.linspace(0, 2 * np.pi, len(amps))
            y = np.real(amps)
            line.set_data(x, y)
            ax.set_xlim(0, 2 * np.pi)
            plt.pause(0.2)

    for _ in range(steps):
        qbit.mutate(0.01)
        update_plot()
        print(f"Entropy: {qbit.bit_entropy()}")
        print(f"Block size: {qbit.selected_block_size}")
        print(f"Num amplitudes: {len(qbit.amplitudes)}")
        print(f"Block size: {qbit.selected_block_size}")
        print(f"Amplitude range: { np.min(np.real(qbit.amplitudes))}, {np.max(np.real(qbit.amplitudes))}")
        print(f"Amplitude std: {np.std(np.real(qbit.amplitudes))}")


    plt.show()

if __name__ == "__main__":
    main()
