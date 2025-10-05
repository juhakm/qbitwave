"""
Multidimensional Dynamic Extension of QBitwave
==============================================

The **QBitwaveND** class generalizes the informational QBitwave framework
into an N-dimensional, continuous (field-like) form. Whereas ``QBitwave``
extracts an emergent *wavefunction* ψ from a primitive bitstring by
optimal compression, **QBitwaveND** takes such an amplitude field and
evolves it dynamically in time.

Conceptually:
    • QBitwave  →  Emergence:  bitstring → ψ(x)
    • QBitwaveND →  Evolution:  ψ(x) → ψ(x, t)

Together they form a complete informational pipeline:
the raw data defines the initial quantum-like state,
and QBitwaveND applies physically motivated, unitary evolution
consistent with the Schrödinger free-particle dispersion relation.

Mathematical foundation
------------------------
Given an N-dimensional complex amplitude array ψ(x₁, x₂, …, xₙ),
QBitwaveND computes its Fourier transform:
    ψ̃(k) = FFT[ψ(x)] / ∏ shape

Time evolution is performed in frequency space as:
    ψ̃(k, t) = ψ̃(k) · exp(-i·ω(k)·t)
where ω(k) = (ħ |k|²) / (2m)

The inverse transform yields ψ(x, t), which can be evaluated at
arbitrary coordinates using Fourier synthesis. This corresponds
to the free-particle solution of the Schrödinger equation in N
dimensions, but here it is framed informationally rather than
ontologically: time evolution is simply *phase evolution in the
optimal compression domain*.


Attributes
----------
amplitudes : np.ndarray
    Complex N-dimensional amplitude field ψ(x) at t=0.
shape : tuple[int]
    Spatial dimensions of the amplitude array.
ndim : int
    Number of spatial dimensions.
fft_coeffs : np.ndarray
    Normalized Fourier coefficients ψ̃(k) of the amplitude field.
freqs : list[np.ndarray]
    Per-axis frequency arrays (in cycles per unit length).
mass : float
    Effective mass parameter used in ω(k) = ħk² / 2m.
c : float
    Speed of light (for optional relativistic corrections).
hbar : float
    Reduced Planck constant, controlling dispersion scaling.

Methods
-------
from_array(data_array):
    Construct a QBitwaveND from a given complex N-D array.
from_qbitwave(qb: QBitwave):  (recommended addition)
    Create a QBitwaveND instance from a 1D informational wavefunction.
time_evolve_coeffs(t):
    Return Fourier coefficients after time evolution e^{-iωt}.
evaluate(*coords, t=0.0):
    Compute ψ(x, t) at arbitrary spatial coordinates.
probability(*coords, t=0.0):
    Return |ψ(x, t)|² at given coordinates (Born rule analog).

Interpretation
---------------
• QBitwaveND treats time as an informational parameter — not as a
    background dimension but as the *phase evolution of encoded structure*.
• This class provides a measurable link between algorithmic information
    (Kolmogorov domain) and spacetime dynamics (Fourier domain).
• In this sense, it realizes a **unitary time evolution over emergent
    informational geometry**, extending the static ψ of QBitwave into
    a living ψ(x, t).


Example
-------
>>> qb = QBitwave("010110110001")
>>> qn = QBitwaveND.from_qbitwave(qb)
>>> ψ_t = qn.evaluate(0.2, t=0.5)
>>> P = qn.probability(0.2, t=0.5)


Copyright
---------
(c) 2019–2025 Juha Meskanen
All rights reserved.
"""


from typing import Optional, Tuple, Union
import numpy as np
import warnings
from numba import njit, prange
from qbitwave import QBitwave

class QBitwaveND(QBitwave):
    def __init__(self, data, mass=1.0, c=1.0, hbar=1.0):
        import numpy as np
        from qbitwave import QBitwave

        # Convert string or list of bits to a 1D normalized amplitude array
        if isinstance(data, str):
            super().__init__(data)  # initialize QBitwave core
            bit_list = [int(b) for b in data if b in "01"]
            if len(bit_list) == 0:
                raise ValueError("Cannot initialize from empty string")
            q = QBitwave(bit_list)
            amplitudes = q.get_amplitudes()
        elif isinstance(data, list):
            # assume list of numbers or bits
            if all(isinstance(b, (int, np.integer)) for b in data):
                q = QBitwave(data)
                amplitudes = q.get_amplitudes()
            else:
                amplitudes = np.array(data, dtype=np.complex64)
        elif isinstance(data, np.ndarray):
            amplitudes = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Ensure amplitudes is a NumPy array of complex type
        self.amplitudes = np.array(amplitudes, dtype=np.complex64)

        # Record shape and dimensions
        self.shape = self.amplitudes.shape
        self.ndim = len(self.shape)

        # Compute normalized FFT coefficients
        self.fft_coeffs = np.fft.fftn(self.amplitudes) / np.prod(self.shape)
        # Frequencies along each axis
        self.freqs = [np.fft.fftfreq(n) for n in self.shape]

        # Physical parameters
        self.mass = mass
        self.c = c
        self.hbar = hbar

    @classmethod
    def from_array(cls, data_array):
        return cls(data_array)
    
    def time_evolve_coeffs(self, t):
        """
        Compute time-evolved Fourier coefficients at time t,
        applying unitary evolution with dispersion ω(k).
        """
        if t == 0:
            return self.fft_coeffs
        
        # Build frequency grids (convert to angular frequency: 2π * f)
        freq_grids = np.meshgrid(*self.freqs, indexing='ij')
        k_squared = np.zeros_like(freq_grids[0], dtype=np.float64)
        for k in freq_grids:
            # Multiply each frequency by 2π to get the true angular value
            k_squared += (2 * np.pi * k) ** 2
        omega = (self.hbar * k_squared) / (2 * self.mass)
        
        # Time evolution phase factor
        phase_factor = np.exp(-1j * omega * t)
        
        # Apply phase factor to FFT coefficients
        return self.fft_coeffs * phase_factor

    def evaluate(self, *coords, t=0.0, coeffs=None):
        if len(coords) != self.ndim:
            raise ValueError(f"Expected {self.ndim} coordinates, got {len(coords)}")
        
        # Here we treat the passed coordinates as physical coordinates.
        # (Remove or modify the mapping below if you want a different scaling.)
        pos = np.array(coords)  # Use coordinates as provided
        
        # Build frequency grids using explicit angular frequencies,
        # so that each frequency is given by k = 2π * fftfreq(n)
        k_lists = [2 * np.pi * np.fft.fftfreq(n) for n in self.shape]
        k_grids = np.meshgrid(*k_lists, indexing='ij')
        
        # Compute continuous phase = sum_i (k_i * pos_i)
        phase = np.zeros_like(self.fft_coeffs, dtype=np.float64)
        for i in range(self.ndim):
            phase += k_grids[i] * pos[i]
        
        # Get time-evolved Fourier coefficients (if not pre-supplied)
        if coeffs is None:
            coeffs = self.time_evolve_coeffs(t)
        
        # Fourier synthesis: sum_k coeff_k * exp(i * phase)
        # (Since phase is already formed using 2π, we no longer multiply by 2jπ here.)
        val = np.sum(coeffs * np.exp(1j * phase))
        
        return val

    def probability(self, *coords: float, t: float = 0.0, coeffs=None) -> float:
        val = self.evaluate(*coords, t=t, coeffs=coeffs)
        return np.abs(val) ** 2  

    def mutate(self, level: float = 0.01, compressibility_bias: bool = True) -> None:
            """
            Mutate the amplitudes with optional compressibility bias.

            Parameters
            ----------
            level : float
                Fractional strength of mutation (0 = no change, 1 = large mutation)
            compressibility_bias : bool
                If True, apply a low-pass filter in frequency space to favor smooth
                (compressible) configurations.
            """
            # Step 1: Add complex Gaussian noise
            noise = (np.random.randn(*self.amplitudes.shape) +
                    1j * np.random.randn(*self.amplitudes.shape))
            self.amplitudes += level * noise

            # Step 2: Apply compressibility bias if requested
            if compressibility_bias:
                fft_coeffs = np.fft.fftn(self.amplitudes)
                grids = np.meshgrid(*[np.fft.fftfreq(n) for n in self.shape], indexing='ij')
                k_squared = np.zeros_like(grids[0])
                for k in grids:
                    k_squared += k**2
                # Gaussian low-pass: higher k (less compressible) is damped
                sigma = 0.1  # adjust for bias strength
                bias_filter = np.exp(-k_squared / (2 * sigma**2))
                fft_coeffs *= bias_filter
                self.amplitudes = np.fft.ifftn(fft_coeffs)

            # Step 3: Renormalize to unit L2 norm
            norm = np.linalg.norm(self.amplitudes)
            if norm > 0:
                self.amplitudes /= norm