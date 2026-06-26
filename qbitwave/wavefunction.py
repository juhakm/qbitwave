"""Spectral Complexity measure for complex-valued wavefunctions.

Theory background (Meskanen 2026 — "The Wavefunction as Compression")
======================================================================
The central hypothesis is that the quantum wavefunction is the universe's
data-compression codec.  Internal observers — themselves composed of
compressed structures — perceive their constituent degrees of freedom as
wave-like because they are observing *compressed information*.  The codec
that produces this compression is the Fourier / spectral decomposition.

Spectral Complexity C_s
-----------------------
A wavefunction ψ(x) can always be written as a superposition of spectral
modes, each characterised by two attributes:

    frequency  ω  — the rate of oscillation, unbounded above zero
    phase      φ  — the offset of the oscillation, bounded in [0, 2π)

The *spectral complexity* C_s(ψ) is the total continuous information cost
needed to specify the set of modes that materially compose ψ:

    C_s(ψ) = Σ_i  [ ω_i / Δω  +  φ_cost(φ_i) ]

Frequency cost (dominant term)
    ω_i / Δω is the number of resolution steps Δω needed to locate
    frequency ω_i.  It is unbounded, continuous, and grows linearly with
    frequency.  This term *dominates* C_s and is the reason the measure
    exponentially suppresses high-frequency (rough, chaotic) states.
    The identification Δω = ℏ ln 2 connects the minimum frequency
    resolution to Planck's constant (Meskanen 2026, §3.1).

Phase cost (subdominant, bounded)
    Each phase φ_i ∈ [0, 2π) requires a finite amount of information to
    specify.  The cost is *global* over all modes: it measures how much
    information is needed to distinguish the phases from one another.
    With only two modes at phases 0 and π, very little is needed; with
    many modes at crowded, uneven phases, somewhat more is required.
    In practice this term is bounded by log₂(N_modes) and is a
    second-order correction.  The current implementation uses a simple
    uniform fixed cost per non-reference mode as a tractable proxy; the
    reference mode (highest amplitude) is exempt because only *relative*
    phases are observable — a global phase shift leaves |ψ(x)|² unchanged.

Amplitude and the fidelity engine
    Amplitude does not appear as a separate encoding cost.  Instead it
    determines *which modes are included* in the description via a
    power-ranked fidelity engine: modes are added in descending power
    order until the accumulated power reaches a target fraction of the
    total.  Modes below this threshold are simply absent from the
    description — they are not part of the codec output and contribute
    zero complexity cost.  This correctly handles the case where many
    weak modes coexist with a few dominant ones: the dominant modes
    determine C_s; the weak modes are free.

Solomonoff suppression and the probability profile
    Under Solomonoff-like induction the prior probability of a
    configuration is P(ψ) ∝ 2^{-C_s(ψ)}.  Because C_s is a *sum* over
    independent modes, the probability *factorises*:

        P(ψ) ∝ Π_i  2^{-ω_i/Δω}

    Each mode is suppressed independently and exponentially by its
    frequency.  The resulting probability profile is Boltzmann-like with
    inverse temperature β = ln(2)/Δω:

        P(mode i present) ∝ exp(−ln(2) · ω_i / Δω)

    Smooth, low-frequency, compressible states dominate the measure.
    Boltzmann brains, random fluctuations, and chaotic configurations
    are exponentially suppressed — not by fine-tuning, but because they
    require many high-frequency modes to describe.

The conjecture C_s ∝ S_Euclidean
    The central open conjecture (Meskanen 2026, §4) is that the minimum
    spectral complexity path through configuration space coincides with
    the minimum Euclidean action path of standard quantum gravity.  If
    true, quantum gravity is Solomonoff induction over compressed
    descriptions of geometry, and ℏ is the minimum spectral resolution
    of a finite informational universe.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Spectral mode
# ---------------------------------------------------------------------------

@dataclass
class SpectralMode:
    """A single Fourier mode in the spectral decomposition of a wavefunction.

    Attributes:
        frequency: Angular frequency ω (rad / spatial-unit).  Signed —
            positive and negative frequencies are physically distinct for
            complex ψ.  The frequency cost uses |ω|.
        amplitude: Real-valued mode amplitude A ≥ 0, equal to
            |FFT_k| / N after normalisation.  Used only by the fidelity
            engine to rank modes by power; it does not appear in C_s.
        phase: Mode phase φ ∈ [0, 2π), equal to arg(FFT_k) mod 2π.
            Relative phases between modes determine interference structure
            and enter C_s through the (subdominant) phase cost term.
    """

    frequency: float
    amplitude: float
    phase: float

    def __post_init__(self) -> None:
        self.amplitude = float(abs(self.amplitude))
        self.phase = float(self.phase % (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Wavefunction
# ---------------------------------------------------------------------------

class Wavefunction:
    """A normalised complex wavefunction on a uniform 1-D spatial grid.

    The wavefunction ψ(x) is stored as a complex NumPy array of length N.
    Its primary analytical interface is the spectral complexity C_s(ψ),
    a continuous, computable measure of how many spectral resources are
    required to describe ψ to a given fidelity.

    Spectral complexity formula
    ---------------------------
    C_s(ψ) = Σ_{i ∈ retained}  [ |ω_i| / Δω  +  φ_cost(i) ]

    where the sum runs over modes retained by the fidelity engine
    (see below), and:

        |ω_i| / Δω   — frequency cost: dominant, continuous, unbounded.
                        Mirrors E = ℏω; under Solomonoff suppression
                        2^{-C_s} this produces an exponential / Boltzmann
                        distribution over frequency.

        φ_cost(i)     — phase cost: subdominant, bounded.  Zero for the
                        reference mode (highest amplitude — global phase
                        unobservable); a fixed resolution cost
                        `phase_resolution` for every other retained mode.
                        Represents the information needed to distinguish
                        this mode's phase from the reference.

    Fidelity engine
    ---------------
    Not all FFT bins materially contribute to ψ.  The fidelity engine
    ranks bins by power (|FFT_k|²) and greedily accumulates them until
    the retained set accounts for at least `fidelity_target` of the
    total power.  Only retained modes enter C_s.  This has two effects:

    1.  Correctness: two wavefunctions with the same |ψ(x)|² but
        different numbers of negligible-power modes receive the same C_s.
        The codec describes the *observable state*, not a particular
        representation.

    2.  Stability: noise and numerical artefacts in low-power bins do
        not inflate C_s.

    Planck identification
    ---------------------
    The minimum resolvable frequency Δω defaults to 2π / (N dx), the
    natural FFT resolution of the grid.  The identification
    ℏ = Δω / ln(2) connects this resolution to Planck's constant
    (Meskanen 2026, §3.1).  This is a correspondence to be investigated,
    not a derivation.

    Args:
        psi: Complex wavefunction amplitudes on a uniform spatial grid.
            Normalised to unit L² norm on construction.
        dx: Spatial grid spacing.  Determines angular frequencies via
            ω_k = 2π k / (N dx).  Defaults to 1.0.
        delta_omega: Minimum resolvable angular frequency Δω.  Defaults
            to the natural FFT resolution 2π / (N dx).
        fidelity_target: Fraction of total spectral power that the
            retained modes must collectively explain.  Must be in (0, 1].
            Higher values retain more modes and increase C_s.
            Defaults to 0.999 (99.9 % of power).
        phase_resolution: Continuous cost assigned to the phase of each
            non-reference retained mode.  Represents the information
            needed to resolve that mode's phase from the reference to
            the required precision.  Bounded and subdominant relative to
            the frequency cost.  Defaults to 1.0 (one unit per mode,
            interpretable as ~ 1 bit of phase separation information).

    Attributes:
        dx: Spatial grid spacing.
        delta_omega: Minimum resolvable angular frequency Δω.
        fidelity_target: Power-fidelity threshold for mode retention.
        phase_resolution: Per-mode phase cost for non-reference modes.
        hbar_identified: Δω / ln(2) — the value of ℏ implied by the
            grid resolution under the Meskanen (2026) identification.
    """

    def __init__(
        self,
        psi: np.ndarray,
        dx: float = 1.0,
        delta_omega: Optional[float] = None,
        fidelity_target: float = 0.999,
        phase_resolution: float = 1.0,
    ) -> None:
        self._psi: np.ndarray = self._normalise(np.asarray(psi, dtype=complex))
        self.dx: float = float(dx)
        self.fidelity_target: float = float(fidelity_target)
        self.phase_resolution: float = float(phase_resolution)

        N = len(self._psi)
        self.delta_omega: float = (
            float(delta_omega) if delta_omega is not None
            else 2.0 * np.pi / (N * self.dx)
        )
        self.hbar_identified: float = self.delta_omega / np.log(2.0)

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def psi(self) -> np.ndarray:
        """Normalised complex wavefunction array ψ(x), shape (N,)."""
        return self._psi

    @property
    def N(self) -> int:
        """Number of spatial grid points."""
        return len(self._psi)

    @property
    def x(self) -> np.ndarray:
        """Spatial grid positions x_n = n · dx, shape (N,)."""
        return np.arange(self.N) * self.dx

    @property
    def probability_density(self) -> np.ndarray:
        """|ψ(x)|² — Born-rule probability density, shape (N,).

        Normalised so that sum(|ψ|²) = 1 (discrete L² norm).
        """
        return np.abs(self._psi) ** 2

    # ------------------------------------------------------------------
    # Fidelity engine — mode selection
    # ------------------------------------------------------------------

    def retained_modes(self) -> list[SpectralMode]:
        """Return the set of spectral modes retained by the fidelity engine.

        Modes are ranked by power (|FFT_k|²) in descending order and
        accumulated greedily until the retained set accounts for at least
        `fidelity_target` of the total power.  The result is the minimal
        set of modes that reproduces ψ(x) to the required fidelity.

        The reference mode (highest power, first in the ranked list) is
        flagged implicitly: its phase is the global reference and costs
        nothing.  All other retained modes pay `phase_resolution`.

        Returns:
            List of SpectralMode objects sorted by ascending |ω|, one per
            retained FFT bin.  The highest-power mode appears somewhere
            in this list and is identifiable as the one with the largest
            amplitude.
        """
        N = self.N
        fft_coeffs: np.ndarray = np.fft.fft(self._psi)
        freqs: np.ndarray = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        power: np.ndarray = np.abs(fft_coeffs) ** 2
        total_power: float = float(power.sum())

        if total_power == 0.0:
            return []

        # Rank bins by descending power
        sorted_idx: np.ndarray = np.argsort(power)[::-1]

        accumulated: float = 0.0
        kept: list[int] = []
        for idx in sorted_idx:
            accumulated += float(power[idx])
            kept.append(int(idx))
            if accumulated / total_power >= self.fidelity_target:
                break

        modes: list[SpectralMode] = [
            SpectralMode(
                frequency=float(freqs[k]),
                amplitude=float(np.abs(fft_coeffs[k])) / N,
                phase=float(np.angle(fft_coeffs[k]) % (2.0 * np.pi)),
            )
            for k in kept
        ]
        modes.sort(key=lambda m: abs(m.frequency))
        return modes

    # ------------------------------------------------------------------
    # Spectral complexity  C_s(ψ)
    # ------------------------------------------------------------------

    def spectral_complexity(self, verbose: bool = False) -> float:
        """Compute the spectral complexity C_s(ψ).

        C_s is the total continuous information cost of specifying the
        retained spectral modes:

            C_s(ψ) = Σ_{i ∈ retained} [ |ω_i| / Δω  +  φ_cost(i) ]

        where φ_cost(i) = 0 for the reference mode (highest amplitude)
        and `phase_resolution` for all other retained modes.

        The frequency term |ω_i| / Δω is the dominant, unbounded
        contribution.  The phase term is a bounded correction that
        accounts for the information needed to distinguish each mode's
        phase from the reference.

        Under Solomonoff suppression P(ψ) ∝ 2^{-C_s(ψ)}, and because
        the sum factorises over modes:

            P(ψ) ∝ Π_i  2^{-|ω_i|/Δω}  ·  2^{-φ_cost(i)}

        The probability profile is exponential in frequency — each mode
        independently suppressed by its oscillation rate.

        Args:
            verbose: If True, print a per-mode breakdown to stdout.

        Returns:
            C_s as a non-negative float.  Zero only if no modes are
            retained (degenerate zero wavefunction).
        """
        modes: list[SpectralMode] = self.retained_modes()
        if not modes:
            return 0.0

        # Reference mode: highest amplitude — pays zero phase cost.
        # Only relative phases are observable; the highest-amplitude mode
        # is the natural phase reference.  Ties broken by lowest |ω| for
        # numerical stability (avoids C_s jumping by 1 when two equal-power
        # modes exchange amplitudes under tiny perturbations).
        ref_amplitude: float = max(m.amplitude for m in modes)
        ref_freq_abs: float = min(
            abs(m.frequency) for m in modes if m.amplitude == ref_amplitude
        )

        total: float = 0.0

        if verbose:
            print(f"\n  {'i':>4}  {'ω':>10}  {'A':>9}  {'φ/2π':>7}  "
                  f"{'|ω|/Δω':>10}  {'φ_cost':>7}  {'mode C_s':>9}")
            print("  " + "-" * 62)

        for i, mode in enumerate(modes):
            freq_cost: float = abs(mode.frequency) / self.delta_omega
            # Reference mode (global phase) is unobservable — zero phase cost
            is_reference: bool = (
                mode.amplitude == ref_amplitude
                and abs(mode.frequency) == ref_freq_abs
            )
            phase_cost: float = 0.0 if is_reference else self.phase_resolution
            mode_cs: float = freq_cost + phase_cost
            total += mode_cs

            if verbose:
                print(f"  {i:>4}  {mode.frequency:>10.4f}  {mode.amplitude:>9.5f}"
                      f"  {mode.phase / (2*np.pi):>7.4f}"
                      f"  {freq_cost:>10.3f}  {phase_cost:>7.2f}  {mode_cs:>9.3f}"
                      + ("  ← ref" if is_reference else ""))

        if verbose:
            print("  " + "-" * 62)
            n_ret = len(modes)
            print(f"  C_s = {total:.4f}   "
                  f"({n_ret} modes retained at fidelity ≥ {self.fidelity_target})\n")

        return total

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def solomonoff_weight(self) -> float:
        """Unnormalised Solomonoff prior weight 2^{-C_s(ψ)}.

        Under Solomonoff-like induction, configurations are weighted by
        their compressibility.  The weight is exponential in C_s:

            w(ψ) = 2^{-C_s(ψ)}

        Because C_s factorises over modes, this equals the product of
        per-mode suppression factors:

            w(ψ) = Π_i  2^{-|ω_i|/Δω}  ·  2^{-φ_cost(i)}

        Smooth, low-frequency wavefunctions receive exponentially higher
        weight than chaotic, high-frequency ones.

        Returns:
            Float in (0, 1].  Returns 1.0 for the zero wavefunction
            (no retained modes, C_s = 0).
        """
        return 2.0 ** (-self.spectral_complexity())

    def mode_suppression_factors(self) -> dict[float, float]:
        """Per-mode Boltzmann suppression factors from the frequency cost.

        For each retained mode i, the frequency term |ω_i| / Δω
        contributes an independent suppression factor:

            s_i = 2^{-|ω_i|/Δω} = exp(−ln2 · |ω_i| / Δω)

        This is a Boltzmann factor with inverse temperature
        β = ln(2) / Δω.  The identification ℏ = Δω / ln(2) makes
        β = 1 / ℏ in natural units, consistent with E = ℏω.

        Returns:
            Dict mapping ω_i (float) → unnormalised suppression factor s_i.
            The factors are *not* normalised to a probability distribution
            here; use solomonoff_weight() for the joint weight of the full
            wavefunction.
        """
        return {
            m.frequency: 2.0 ** (-abs(m.frequency) / self.delta_omega)
            for m in self.retained_modes()
        }

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def gaussian_packet(
        cls,
        N: int = 256,
        x0: float = 0.0,
        sigma: float = 1.0,
        k0: float = 1.0,
        dx: float = 0.1,
        **kwargs,
    ) -> "Wavefunction":
        """Construct a Gaussian wave packet.

        ψ(x) = exp(−(x−x0)² / 4σ²) · exp(i k0 x),  then normalised.

        A Gaussian packet is narrow in both position and momentum space
        and serves as a natural low-C_s reference state: it has a single
        dominant frequency k0 and a smooth Gaussian envelope that
        concentrates spectral power near k0.

        Args:
            N: Number of grid points.
            x0: Packet centre in position space.
            sigma: Position-space width (standard deviation).
            k0: Carrier wavenumber (dominant angular frequency ω ≈ k0).
            dx: Spatial grid spacing.
            **kwargs: Passed to Wavefunction.__init__
                (e.g. fidelity_target, phase_resolution).

        Returns:
            Normalised Wavefunction instance.
        """
        x = np.arange(N) * dx
        psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
        return cls(psi, dx=dx, **kwargs)

    @classmethod
    def plane_wave_superposition(
        cls,
        N: int = 256,
        amplitudes: list[float] = (0.6, 0.8),
        wavenumbers: list[float] = (1.0, 3.0),
        phases: list[float] = (0.0, 0.0),
        dx: float = 0.1,
        **kwargs,
    ) -> "Wavefunction":
        """Construct an explicit superposition of plane waves.

        ψ(x) = Σ_i  α_i · exp(i (k_i x + φ_i)),  then normalised.

        Useful for constructing states with known spectral structure and
        testing how C_s responds to adding modes or shifting phases.

        Args:
            N: Number of grid points.
            amplitudes: Real amplitudes α_i for each plane wave.
            wavenumbers: Wavenumbers k_i (angular frequencies ω_i = k_i).
            phases: Initial phases φ_i for each plane wave.
            dx: Spatial grid spacing.
            **kwargs: Passed to Wavefunction.__init__.

        Returns:
            Normalised Wavefunction instance.
        """
        x = np.arange(N) * dx
        psi = np.zeros(N, dtype=complex)
        for a, k, phi in zip(amplitudes, wavenumbers, phases):
            psi += a * np.exp(1j * (k * x + phi))
        return cls(psi, dx=dx, **kwargs)

    @classmethod
    def random_state(
        cls,
        N: int = 256,
        seed: int = 42,
        dx: float = 0.1,
        **kwargs,
    ) -> "Wavefunction":
        """Construct a maximally chaotic (high-C_s) random wavefunction.

        Draws real and imaginary parts independently from N(0,1).  The
        resulting state activates all FFT bins with roughly equal power,
        giving the maximum number of retained modes and hence the highest
        possible C_s for a given N.  Useful as an upper-bound reference.

        Args:
            N: Number of grid points.
            seed: Random seed for reproducibility.
            dx: Spatial grid spacing.
            **kwargs: Passed to Wavefunction.__init__.

        Returns:
            Normalised Wavefunction instance.
        """
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        return cls(psi, dx=dx, **kwargs)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Wavefunction") -> "Wavefunction":
        """Superpose two wavefunctions: ψ_new = normalise(ψ_a + ψ_b).

        The result is renormalised.  Interference fringes in the
        superposition share spectral modes, so C_s(ψ_a + ψ_b) is
        typically less than C_s(ψ_a) + C_s(ψ_b) — the codec description
        of the superposition is cheaper than two independent descriptions.

        Args:
            other: Wavefunction on the same grid (same N and dx).

        Returns:
            New normalised Wavefunction.

        Raises:
            ValueError: If grid sizes differ.
        """
        if self.N != other.N:
            raise ValueError(
                f"Grid size mismatch: {self.N} vs {other.N}")
        return Wavefunction(
            self._psi + other._psi,
            dx=self.dx,
            delta_omega=self.delta_omega,
            fidelity_target=self.fidelity_target,
            phase_resolution=self.phase_resolution,
        )

    def __mul__(self, scalar: complex) -> "Wavefunction":
        """Scale ψ by a complex scalar (result is renormalised)."""
        return Wavefunction(
            self._psi * scalar,
            dx=self.dx,
            delta_omega=self.delta_omega,
            fidelity_target=self.fidelity_target,
            phase_resolution=self.phase_resolution,
        )

    def __rmul__(self, scalar: complex) -> "Wavefunction":
        return self.__mul__(scalar)

    def inner_product(self, other: "Wavefunction") -> complex:
        """Discrete inner product ⟨self|other⟩ = Σ_x ψ*(x) φ(x) dx.

        Args:
            other: Wavefunction on the same grid.

        Returns:
            Complex scalar.
        """
        return complex(np.sum(self._psi.conj() * other._psi) * self.dx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(psi: np.ndarray) -> np.ndarray:
        """Normalise ψ to unit L² norm.

        Args:
            psi: Complex array.

        Returns:
            ψ / ‖ψ‖.

        Raises:
            ValueError: If ψ is identically zero.
        """
        norm: float = float(np.sqrt(np.sum(np.abs(psi) ** 2)))
        if norm == 0.0:
            raise ValueError("Cannot normalise a zero wavefunction.")
        return psi / norm

    def __repr__(self) -> str:
        cs = self.spectral_complexity()
        n_modes = len(self.retained_modes())
        return (
            f"Wavefunction(N={self.N}, dx={self.dx}, "
            f"C_s={cs:.3f}, modes={n_modes}, "
            f"ℏ_id={self.hbar_identified:.3e})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(__doc__)
    print("=" * 65)
    print("Demo")
    print("=" * 65)

    # 1. Gaussian packet — few dominant modes, low C_s
    gp = Wavefunction.gaussian_packet(N=256, x0=12.8, sigma=2.0, k0=2.0, dx=0.1)
    print("\n[1] Gaussian wave packet")
    gp.spectral_complexity(verbose=True)
    print(gp)

    # 2. Two-component superposition — moderate C_s
    sp = Wavefunction.plane_wave_superposition(
        N=256, amplitudes=[0.6, 0.8],
        wavenumbers=[1.0, 5.0], phases=[0.0, np.pi / 4], dx=0.1)
    print("\n[2] Two-component superposition")
    sp.spectral_complexity(verbose=True)
    print(sp)

    # 3. Random state — all modes active, high C_s
    rnd = Wavefunction.random_state(N=256, seed=0, dx=0.1)
    print("\n[3] Random (high-entropy) state")
    print(f"  C_s = {rnd.spectral_complexity():.2f}  (verbose suppressed)")
    print(rnd)

    # 4. Solomonoff weights — exponential ordering
    print("\nSolomonoff weights  2^{{-C_s}}:")
    for label, wf in [("Gaussian", gp), ("Superposition", sp), ("Random", rnd)]:
        print(f"  {label:>16s}:  {wf.solomonoff_weight():.4e}")

    # 5. Interference economy
    gp2 = Wavefunction.gaussian_packet(N=256, x0=20.0, sigma=2.0, k0=2.0, dx=0.1)
    combined = gp + gp2
    print(f"\n[5] Interference economy:")
    print(f"  C_s(ψ₁)       = {gp.spectral_complexity():.2f}")
    print(f"  C_s(ψ₂)       = {gp2.spectral_complexity():.2f}")
    print(f"  C_s(ψ₁ + ψ₂)  = {combined.spectral_complexity():.2f}")
    print("  Superposition is cheaper than two independent descriptions.")

    # 6. Fidelity engine: weak modes don't inflate C_s
    x = np.arange(64)
    psi_clean = sum(np.exp(1j * 2*np.pi*k*x/64) for k in [2, 7, 13])
    rng = np.random.default_rng(0)
    psi_noisy = psi_clean + 0.01 * sum(
        np.exp(1j * (2*np.pi*k*x/64 + rng.uniform(0, 2*np.pi)))
        for k in range(20, 50))
    wf_clean = Wavefunction(psi_clean, dx=0.1)
    wf_noisy = Wavefunction(psi_noisy, dx=0.1)
    print(f"\n[6] Fidelity engine — weak modes ignored:")
    print(f"  C_s(clean, 3 modes)          = {wf_clean.spectral_complexity():.3f}")
    print(f"  C_s(+ 30 weak noise modes)   = {wf_noisy.spectral_complexity():.3f}")
    print(f"  Modes retained (clean/noisy) = "
          f"{len(wf_clean.retained_modes())} / {len(wf_noisy.retained_modes())}")
