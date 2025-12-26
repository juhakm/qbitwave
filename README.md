# QBitwave: Emergent Information-Theoretic Wavefunctions

QBitwave models quantum-like dynamics as the deterministic evolution of compressibility in finite bitstrings.
The wavefunction ψ is interpreted not as a physical field but as the minimal compression algorithm that
reproduces a given informational state.
Existence corresponds to self-compressibility — only compressible configurations persist.
Through Fourier-domain transformations and entropy measures, QBitwave unifies bitstrings, complex amplitudes, and probabilistic
behavior into a single information-centric framework.


## Core Concept

### QBitwave

`QBitwave` treats the wavefunction as an **emergent, information-theoretic object**.  
A finite bitstring encodes a discretized wavefunction, which can be reconstructed as normalized complex amplitudes — the **minimal program
reproducing the bitstring** (Kolmogorov complexity perspective).

**Fundamental principles:**

- Compression = quantum probability amplitude = predictability
- Smooth, regular data compresses well → high amplitude in few Fourier components (low entropy)
- Random/noisy data is incompressible → low amplitude concentration
- Wavefunction = minimal program reproducing the bitstring

**Features:**

- Forward mapping: wavefunction → bitstring
- Reverse mapping: bitstring → minimal wavefunction
- Block-size selection via entropy maximization
- Shannon entropy computation
- Fourier-based compressibility measure reflecting structure


### QBitwaveND

`QBitwaveND` generalizes `QBitwave` to **N-dimensional continuous fields** and allows **dynamical evolution in time**.  

| Conceptual Relation |
|--------------------|
| `QBitwave` → Emergence: bitstring → ψ(x) |
| `QBitwaveND` → Evolution: ψ(x) → ψ(x, t) |

`QBitwaveND` applies unitary, physically motivated evolution consistent with the **Schrödinger free-particle dispersion relation**, but framed entirely **informationally**:

1. Take N-dimensional complex amplitude array ψ(x₁, x₂, …, xₙ)
2. Compute Fourier transform:  
   ψ̃(k) = FFT[ψ(x)] / ∏ shape
3. Apply time evolution in frequency space:  
   ψ̃(k, t) = ψ̃(k) · exp(-i·ω(k)·t), where ω(k) = (ħ |k|²) / 2m
4. Inverse transform to get ψ(x, t)

**Interpretation:**

- Time is an informational parameter — the **phase evolution of encoded structure**
- Bridges algorithmic information (Kolmogorov domain) and spacetime dynamics (Fourier domain)
- Provides **unitary time evolution over emergent informational geometry**, extending static ψ(x) of `QBitwave` to ψ(x, t)

**Attributes:**

- `amplitudes` : N-dimensional complex array ψ(x) at t=0  
- `shape` : spatial dimensions of the array  
- `ndim` : number of spatial dimensions  
- `fft_coeffs` : normalized Fourier coefficients ψ̃(k)  
- `freqs` : per-axis frequency arrays  
- `mass` : effective mass parameter (ħk² / 2m)  
- `c` : speed of light (for optional relativistic corrections)  
- `hbar` : reduced Planck constant

**Key Methods:**

- `from_array(data_array)` : construct from existing N-D array  
- `from_qbitwave(qb: QBitwave)` : create N-D field from a 1D informational wavefunction  
- `time_evolve_coeffs(t)` : return Fourier coefficients after time evolution  
- `evaluate(*coords, t=0.0)` : compute ψ(x, t) at arbitrary coordinates  
- `probability(*coords, t=0.0)` : return |ψ(x, t)|² (Born-rule analog)  

## Why It Matters

- Information as primary ontology: All physical phenomena are encoded by minimal informational descriptions; spacetime, fields, and quantum dynamics are derived, not assumed.
- Compressibility replaces renormalization: High-frequency modes that contribute divergences in conventional QFT are interpreted as incompressible configurations with vanishing physical measure, providing a natural regularization.
- Singularities as structureless limits: Zero execution-trace entropy indicates collapse of all distinguishable geometric degrees of freedom, eliminating the need to treat singularities as breakdowns of physics.

## Source code
- https://github.com/juhakm/qbitwave

## Example Usage

```python
from qbitwave import QBitwave, QBitwaveND

# Create a 1D informational wavefunction
qb = QBitwave("010110110001")

# Lift it to N-dimensional dynamic field
qn = QBitwaveND.from_qbitwave(qb)

# Evaluate amplitude at x=0.2, t=0.5
psi_t = qn.evaluate(0.2, t=0.5)

# Compute probability (Born rule analog)
P = qn.probability(0.2, t=0.5)
```


## Images

<img width="640" height="400" alt="compressibility" src="https://github.com/user-attachments/assets/3086378c-13f8-49b5-9591-6d978713c73f" />
<img width="640" height="400" alt="compressibility_entropy" src="https://github.com/user-attachments/assets/03fc8b75-d9f7-45d7-8348-b5eefacd6d36" />
<img width="640" height="400" alt="photon_heatmap" src="https://github.com/user-attachments/assets/93d06be8-c31b-4a04-b62d-32fad4110f56" />
<img width="640" height="400" alt="qbitwave" src="https://github.com/user-attachments/assets/a40b67c6-7432-4e78-b2a0-18a8e4c0ccd4" />


