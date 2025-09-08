
[![PyPI version](https://badge.fury.io/py/qbitwave.svg)](https://badge.fury.io/py/qbitwave)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/juhakm/qbitwave/actions/workflows/test.yml/badge.svg)](https://github.com/juhakm/qbitwave/actions)


# QBitwave - Discrete Wavefunction.

A class that models a quantum wavefunction as a discrete, information-theoretic structure, where the resolution of the wavefunction and the emergence of physical properties are tied directly to the system's Shannon entropy.

## Description

The DiscreteWavefunction class is founded on the principle that the universe began in a state of zero information and that physical reality—including spacetime and quantum states—is an emergent property of increasing entropy. This class represents a "dithered" view of reality, where the apparent continuity of quantum mechanics and general relativity is an illusion created by a vast, finite number of informational bits.

The core premise is that the resolution of the system is not constant. Instead, it increases with time, corresponding to the universe's total entropy. At the beginning (t=0), the system is in a state of zero bits, perfectly smooth and undefined. As entropy grows, the number of bits increases, allowing for the emergence of complex, superposed microstructures (particles). Our current universe, with its immense entropy, is modeled by a DiscreteWavefunction of such high resolution that its discrete nature is currently beyond our ability to measure directly.

`QBitwave` - a discrete quantum wavefunctions as emergent feature of entropy. Unifies bitstrings, amplitudes, and entropy into a single foundational object.

## Key Concepts

- **Bit = Geometry**: Each bit encodes structure.
- **Bitstring = Universe State**: A single bitstring describes a possible quantum universe.
- **QBitwave = Wavefunction**: A discrete quantum object emerging from discrete information (bitstring).

## Features

- Convert any bitstring into a complex amplitude vector
- Extract |ψ|² and phase 
- Measure entropy and emergent resolution

## Images
![Finite wavefunction](https://github.com/juhakm/qbitwave/blob/main/images/qbitwave.png "Finite wavefunction")


## Example

```python
from qbitwave import QBitwave

bw = QBitwave("1101010111001001")
amplitudes = bw.get_amplitudes()
entropy = bw.entropy()
```


