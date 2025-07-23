# Bitwave

**A minimal informational model of quantum emergence.**

`Bitwave` is a Python library for simulating quantum-like wavefunctions as emergent properties of binary strings. It unifies bitstrings, amplitudes, and entropy into a single foundational object.

## Key Concepts

- **Bit = Geometry**: Each bit encodes structure.
- **Bitstring = Universe State**: A single bitstring describes a possible quantum universe.
- **Bitwave = Wavefunction**: A discrete quantum object derived purely from bits.

## Features

- Convert any bitstring into a complex amplitude vector
- Extract |ψ|² and phase 
- Measure entropy and emergent resolution


## Example

```python
from bitwave import Bitwave

bw = Bitwave("1101010111001001")
amplitudes = bw.get_amplitudes()
entropy = bw.entropy()
