"""
spectral_field_mdl.py

Abstract Spectral Field Model (MDL Framework)
=============================================

This module defines the foundational abstraction for representing
informational fields in a finite spectral domain under a
Minimum Description Length (MDL) principle.

Core Idea
---------
A "field" is not a physical primitive, but a compressed representation
of information accessible to an observer. The observer does not see
reality directly, but instead interacts with a description of reality.

This description is encoded as a finite set of spectral modes over Z_N:

    ψ_k ∈ ℂ^d

where:
    k     : discrete frequency index
    d     : internal dimension of the field (e.g., scalar, spinor, etc.)

The structural complexity of a field is defined as:

    C_Q = Σ (k_eff^2 * ||ψ_k||^2)

where:
    k_eff = min(k, N - k)

This functional measures the roughness or informational cost of the
field. Low-frequency (smooth) descriptions are preferred.

Typicality and Measure
----------------------
The MDL framework assigns a statistical weight to each field:

    P(ψ) ∝ exp(-λ C_Q)

This defines a measure over all possible field configurations, where
more compressible (simpler) descriptions dominate.

Interpretation
--------------
- Geometry, particles, and dynamics are not fundamental.
- They emerge from the competition between compression and distinguishability.
- Internal field dimension (d) corresponds to degrees of freedom such as spin.

This module provides an abstract base class for all such spectral fields.
Concrete implementations include:
    - Scalar fields (d = 1)
    - Spinor fields (d = 2)
    - Higher-dimensional internal symmetry fields (future work)

Design Principles
-----------------
1. The MDL engine is the fundamental layer.
2. All physical structure is derived from spectral compression.
3. FFT-based encoding is centralized and not duplicated.
4. Field representations differ only by internal dimensionality.

Author:
    Juha Meskanen

"""

from abc import ABC, abstractmethod
import numpy as np


class SpectralFieldMDL(ABC):
    """
    Abstract base class for spectral fields under the MDL framework.

    This class defines the interface and shared functionality for all
    informational fields represented in a finite spectral domain.

    Subclasses must implement:
        - Encoding of signals into spectral representation
        - Spectral complexity computation
        - Evaluation (reconstruction) in position space

    Attributes:
        N (int): Size of the discrete domain (Z_N).
    """

    def __init__(self, N: int):
        """
        Initializes the spectral field.

        Args:
            N (int): Size of the discrete domain (number of spatial points).
        """
        self.N = int(N)

    @abstractmethod
    def encode(self, *signals: np.ndarray) -> None:
        """
        Encodes one or more input signals into a spectral representation.

        This method is the canonical entry point for constructing the field
        from observational or simulated data.

        Args:
            *signals (np.ndarray):
                Input signals to encode. The number and interpretation depend
                on the concrete subclass:
                    - Scalar field: one complex signal
                    - Spinor field: two complex signals (alpha, beta)
                    - Higher fields: multiple components

        Raises:
            ValueError: If input signals are invalid or incompatible.
        """
        pass

    @abstractmethod
    def spectral_complexity(self) -> float:
        """
        Computes the structural complexity C_Q of the field.

        The complexity is defined as a frequency-weighted power spectrum:

            C_Q = Σ (k_eff^2 * ||ψ_k||^2)

        where:
            k_eff = min(k, N - k)

        Returns:
            float: The computed spectral complexity.
        """
        pass

    @abstractmethod
    def evaluate(self) -> np.ndarray:
        """
        Reconstructs the field in the position domain.

        This method transforms the spectral representation back into
        the spatial domain (Z_N), producing the observable field.

        Returns:
            np.ndarray:
                Field values in position space. The shape depends on the
                internal dimensionality:
                    - Scalar field: (N,)
                    - Spinor field: (N, 2)
                    - General field: (N, d)
        """
        pass

    def typicality_weight(self, lam: float = 1.0) -> float:
        """
        Computes the typicality weight of the field.

        The weight is defined as:

            P(ψ) = exp(-λ C_Q)

        where:
            λ controls the strength of the compression preference.

        Args:
            lam (float, optional):
                Inverse temperature / compression strength parameter.
                Higher values penalize complexity more strongly.
                Defaults to 1.0.

        Returns:
            float: The statistical weight of the field.
        """
        return np.exp(-lam * self.spectral_complexity())

    @staticmethod
    def bits_estimate(x: float, eps: float = 1e-12) -> float:
        """
        Estimates the description length (in bits) of a real value.

        This provides a logarithmic proxy for encoding cost.

        Args:
            x (float): Value to encode.
            eps (float, optional): Numerical stabilizer. Defaults to 1e-12.

        Returns:
            float: Approximate number of bits required.
        """
        return np.log2(1.0 + abs(x) + eps)

    def description_length_estimate(self) -> float:
        """
        Estimates the total description length of the field.

        This method should be overridden by subclasses to account for
        their specific internal structure (e.g., multiple components).

        Returns:
            float: Estimated number of bits required to encode the field.

        Notes:
            The default implementation returns 0.0 and should be extended
            in concrete subclasses.
        """
        return 0.0
    
