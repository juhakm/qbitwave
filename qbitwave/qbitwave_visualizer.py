import numpy as np
import matplotlib.pyplot as plt
from wavefunction import QBitwave

class QBitwaveVisualizer:
    def __init__(self, wavefunction: QBitwave):
        self.wf = wavefunction
        self.amplitudes = self.wf.get_amplitudes()
        self.indices = np.arange(len(self.amplitudes))
        self.real = np.real(self.amplitudes)
        self.imag = np.imag(self.amplitudes)
        self.prob = np.abs(self.amplitudes) ** 2
        self.phase = np.angle(self.amplitudes)

    def plot_components(self, title: str = "QBitwave Components"):
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        axs[0].bar(self.indices, self.real, color='blue')
        axs[0].set_ylabel("Re(ψ)")
        axs[0].set_title(title)

        axs[1].bar(self.indices, self.imag, color='red')
        axs[1].set_ylabel("Im(ψ)")

        axs[2].bar(self.indices, self.prob, color='black')
        axs[2].set_ylabel("|ψ|² (Prob)")

        axs[3].bar(self.indices, self.phase, color='purple')
        axs[3].set_ylabel("Arg(ψ)")
        axs[3].set_xlabel("State Index")

        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, title: str = "QBitwave Heatmap"):
        data = np.array([self.prob, self.phase])
        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(data, cmap='viridis', aspect='auto')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["|ψ|²", "Phase"])
        ax.set_title(title)

        plt.colorbar(im, ax=ax, orientation='vertical')
        plt.show()
