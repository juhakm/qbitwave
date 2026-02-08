# CHANGELOG.md

## Version 0.2.11 — 2026-02-08

- **wave_complexity():** Upgraded to incorporate phase sensitivity in addition to amplitude/frequency magnitudes.
  Computes the FFT of the full complex wavefunction (preserving phases).
  Spectral entropy is calculated over magnitude squared (power spectral density).
  Phase differences between consecutive spectral components are now measured and contribute to complexity, capturing rapid phase variations.
  Total complexity = magnitude-based spectral entropy + average phase variation.
  Edge cases (empty amplitudes) return 0.0.
  Removed prior Shannon limit assumption from unit tests; wavefunction complexity can exceed naive log₂(number of amplitudes) because many wavefunctions map to the same bitstring.
  This update ensures wave_complexity() aligns with the minimal spectral complexity → maximal predictability principle and properly reflects both amplitude and phase structure.


## Version 0.2.10 — 2026-02-02

- Minor changes to README

## Version 0.2.9 — 2026-01-29

- New method _analyze_bitstring_1to1(self) added

## Version 0.2.8 — 2026-01-24

- Misleading "Kolmogorov complexity" and "minimal program" terminology removed
  and replaced with Nimimal Spectral Length (MSL).


## Version 0.2.6 — **Fixes to docstrings**:

- Obsolete parameter from mutate() method removed
- Wrong type annotations in docstrings fixed.


## Version 0.2.5 — **Improvements**:

- Spectral Transition: In the original code, wave_complexity counted spatial blocks. It now counts Spectral Entropy. 
- Normalization Stability: The compressibility method now uses np.fft.rfft on the real part of the reconstructed amplitudes.


## Version 0.2.4 — **QBitwave.wave_complexity(eps=1e-3)**:
  Computes the spectral complexity of the wavefunction by counting the number of significant amplitude components above a threshold.
  - Provides an intuitive measure of compressibility and algorithmic complexity of a bitstring’s wavefunction representation.
  - Smooth or regular bitstrings yield lower complexity; noisy or irregular bitstrings yield higher complexity.
  - Fully compatible with `QBitwave`'s amplitude normalization and Fourier-based analysis.
  - Returns a native Python `int` for compatibility with unit tests and downstream applications.


## Version 0.2.3 — "Citation.cff added"
**Date:** 2025-12-26
**pyproject.toml:** Keywords updated, version elevated.
**citations.cff:** file added


## Version 0.2.2 — "Wavefunction-Centric Rewrite"
**Date:** 2025-10-19
**Core Concept:** Failing unit test fixed.



## Version 0.2.1 — "Wavefunction-Centric Rewrite"
**Date:** 2025-10-04
**Core Concept:** Quantum behavior emerges statistically from information compressibility.

### Conceptual Shift
- **New model:** `QBitwaveND` treats the *wavefunction itself* as fundamental.
  - The bitstring representation is optional, not required.
  - The system evolves in continuous or discrete N-dimensional configuration space.
  - All physical structure (energy, motion, probability) emerges from transformations in amplitude space.

In short:
> **The wavefunction now *is* the information.**
> Bit patterns are its discrete projections.



### Structural Changes

**Primary Data:**  `amplitudes` (NumPy N-D complex array)
**Core Idea:** Information density patterns form emergent quantum amplitudes
**Dimensionality:** Explicit N-D shape of amplitude array
**Entropy:** Amplitude magnitude statistics via probability density
**Compressibility:** Interpreted as spatial frequency sparsity in Fourier domain
**Mutation / Flip:** Smooth evolution, phase propagation via FFT
**Normalization:** Automatic unitary evolution preserves norm
**Evaluation:**  Continuous evaluation using inverse FFT and phase evolution
**Time Evolution:** Explicit temporal dynamics: `time_evolve_coeffs(t)`
**Representation:** Functional `evaluate(x₁, x₂, ..., t)` returning complex amplitude


### New Features
- **Multi-Dimensional Support:** Works with arbitrary N-dimensional amplitude tensors.
- **Frequency-Domain Representation:** Uses FFT for spatial-temporal evolution.
- **Unitary Time Evolution:** Applies `exp(-i * ω * t)` phase factor to Fourier coefficients.
- **Probabilistic Evaluation:** Returns `|ψ(x,t)|²` as emergent probability density.
- **Mass and Constants:** Optional `mass`, `ħ`, `c` parameters for scaling dynamics.
- **Continuous Coordinates:** Supports floating-point sampling in any spatial dimension.


### Philosophical Notes
- The rewrite aligns with the **information-theoretic hypothesis**:
  > *Quantum probability arises naturally from statistical bias toward compressible configurations.*
- The N-D version **removes all baked-in physics** — no Hamiltonian, no particle postulates.
  Instead, **patterns in informational density** (Fourier sparsity) are treated as emergent “physical laws”.
- The new structure allows viewing any wavefunction as a **minimal program** producing its own informational pattern — satisfying the principle of self-generated reality.


### Testing and Validation
- The old unit tests (`test_qbitwave.py`) relied on bitstring operations.
- A new test suite (`qbitwavend_test.py`) now validates:
  - normalization of FFT coefficients
  - probability non-negativity
  - temporal phase consistency
  - shape and type safety for N-D arrays
- The tests confirm **unitary evolution** and **stable probability evaluation**.


## Version 0.1.0 — ""
**Date:** Jul 23 2025
**Core Concept:** Quantum wavefunction emerging from the given bitstring.
- Features:
  - updated for new Python 3.1
  - type annotations added
  - migrated to pyproject.toml
  - examples/*.py added

### Version 0.0.7 — ""
**Date:** Aug 12 2019
**Core Concept:** Initial release
