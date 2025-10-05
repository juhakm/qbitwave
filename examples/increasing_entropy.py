"""
Wavefunction Emergence from Sine Wave Bitstring
===============================================

Demonstrates how a smooth, low-entropy sine wave bitstring gradually
acquires higher-frequency structure, illustrating the emergence of
complexity in a compressibility-biased system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qbitwave import QBitwave

# Monkey-patch FuncAnimation _stop to not use _resize_id

# monkey patch
def safe_stop(self, *args, **kwargs):
    pass

FuncAnimation._stop = safe_stop

# -----------------------------
# Functions to encode sine waves
# -----------------------------
def amplitude_to_bits(amplitude: float, basis_size: int) -> list[int]:
    max_int = 2**basis_size - 1
    int_val = int(round(amplitude * max_int))
    bits = [(int_val >> i) & 1 for i in reversed(range(basis_size))]
    return bits

def sine_bitstring(num_samples: int, basis_size: int, freqs=[1.0]) -> list[int]:
    t = np.linspace(0, 1, num_samples, endpoint=False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    wave = (wave - wave.min()) / (wave.max() - wave.min())  # normalize to [0,1]
    bitstring = []
    for amp in wave:
        bitstring.extend(amplitude_to_bits(amp, basis_size))
    return bitstring

# -----------------------------
# Simulation parameters
# -----------------------------
num_samples = 256
basis_size = 16
steps = 200
freq_growth = [1]  # start with a single sine component

# -----------------------------
# Initialize QBitwave
# -----------------------------
bitstring = sine_bitstring(num_samples, basis_size, freq_growth)
qbit = QBitwave(bitstring=bitstring, fixed_basis_size=basis_size)

# -----------------------------
# Setup plots
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
line, = ax1.plot([], [], lw=2)
ax1.set_ylim(-0.6, 0.6)
ax1.set_xlim(0, 2 * np.pi)
ax1.set_title("Wavefunction Emergence from Sine Wave")
ax1.set_xlabel("Phase (arbitrary)")
ax1.set_ylabel("Amplitude")

ax2.set_xlim(0, steps)
ax2.set_ylim(0, 10)  # will auto-adjust if needed
bit_entropy_line, = ax2.plot([], [], label="Bit Entropy")
wf_entropy_line, = ax2.plot([], [], label="Wavefunction Entropy")
comp_line, = ax2.plot([], [], label="Compressibility")
ax2.legend()
ax2.set_xlabel("Step")
ax2.set_ylabel("Value")
ax2.set_title("Entropy and Compressibility Over Time")

# -----------------------------
# History trackers
# -----------------------------
bit_entropy_hist = []
wf_entropy_hist = []
comp_hist = []

# -----------------------------
# Animation update function
# -----------------------------
def update(frame):
    global freq_growth

    # Slowly add higher frequencies over time
    if frame % 5 == 0 and len(freq_growth) < 50:
        freq_growth.append(len(freq_growth)+1)  # add new frequency component
        new_bits = sine_bitstring(num_samples, basis_size, freq_growth)
        # qbit.bitstring = new_bits
        qbit.set_bitstring(new_bits)

    # Optional: small random flips for stochasticity
    #qbit.mutate(0.01)

    # Update main waveform
    amps = np.array(qbit.get_amplitudes(), dtype=np.complex128)
    x = np.linspace(0, 2 * np.pi, len(amps))
    y = np.real(amps)
    line.set_data(x, y)

    # Record entropy and compressibility
    bit_entropy_hist.append(qbit.bit_entropy())
    wf_entropy_hist.append(qbit.entropy())
    comp_hist.append(qbit.compressibility())

    bit_entropy_line.set_data(range(len(bit_entropy_hist)), bit_entropy_hist)
    wf_entropy_line.set_data(range(len(wf_entropy_hist)), wf_entropy_hist)
    comp_line.set_data(range(len(comp_hist)), comp_hist)

    # Adjust ax2 limits dynamically
    ax2.set_xlim(0, max(steps, frame+1))
    ax2.set_ylim(0, max(max(bit_entropy_hist), max(wf_entropy_hist), max(comp_hist))+0.5)

    return [line, bit_entropy_line, wf_entropy_line, comp_line]

# -----------------------------
# Run animation
# -----------------------------
anim = FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
plt.tight_layout()
plt.show()
