
[![PyPI version](https://badge.fury.io/py/qbitwave.svg)](https://badge.fury.io/py/qbitwave)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/juhakm/qbitwave/actions/workflows/test.yml/badge.svg)](https://github.com/juhakm/qbitwave/actions)


# QBitwave

**Discrete wavefunction.**

`QBitwave` is a Python library for simulating quantum wavefunctions as emergent properties of binary strings.
It unifies bitstrings, amplitudes, and entropy into a single foundational object.

## Key Concepts

- **Bit = Geometry**: Each bit encodes structure.
- **Bitstring = Universe State**: A single bitstring describes a possible quantum universe.
- **QBitwave = Wavefunction**: A discrete quantum object emerging purely from bits.

## Features

- Convert any bitstring into a complex amplitude vector
- Extract |ψ|² and phase 
- Measure entropy and emergent resolution

## Images
![Finite wavefunction](./images/qbitstr.png "Finite wavefunction")


## Example

```python
from qbitwave import QBitwave

bw = QBitwave("1101010111001001")
amplitudes = bw.get_amplitudes()
entropy = bw.entropy()


