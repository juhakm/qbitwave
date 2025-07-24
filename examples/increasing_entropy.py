import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave


def main():
    steps : int = 500
    num_samples : int = 128
    block_size : int = 16 #precision
    # start out from zero entropy
    bitstring : str = [0] * 1024
    qbit = QBitwave(bitstring, block_size)

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
            plt.pause(0.1)

    for _ in range(steps):
        update_plot()
        print(f"Bitstring entropy: {qbit.bit_entropy()}")
        print(f"Wavefunction entropy: {qbit.entropy()}")
        print(f"Num amplitudes: {len(qbit.amplitudes)}")
        print(f"Amplitude range: { np.min(np.real(qbit.amplitudes))}, {np.max(np.real(qbit.amplitudes))}")
        print(f"Amplitude std: {np.std(np.real(qbit.amplitudes))}")
        qbit.flip()


    plt.show()

if __name__ == "__main__":
    main()
