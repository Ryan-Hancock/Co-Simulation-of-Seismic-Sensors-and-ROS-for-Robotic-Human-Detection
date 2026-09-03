// Package green is the propagation stage: a force applied at the surface
// becomes ground velocity at a receiver some range away.
//
// Slice 0 keeps only the far-field Rayleigh wave over a homogeneous
// half-space. That is a deliberate simplification with a known validity
// limit, and it is worth stating precisely because the limit falls inside the
// range this project cares about: the far-field asymptotic needs r >> lambda,
// and at 20 Hz over soil with cR = 150 m/s the wavelength is 7.5 m. A robot
// detecting a person at 5 m is inside one wavelength, where near-field terms
// and body-wave arrivals are not negligible. Slice 3 replaces this with full
// frequency-wavenumber integration for exactly that reason. Until then, this
// model is honest at 20 m and optimistic at 3 m.
//
// What is exact here: travel time, geometric spreading, the causal pairing of
// attenuation with its dispersion, and the Rayleigh eigenfunctions the
// excitation is built from.
package green

import (
	"fmt"
	"math"
	"math/cmplx"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// Eigenfunctions are the Rayleigh mode shapes of a homogeneous half-space:
// the depth profiles of horizontal and vertical displacement, in the closed
// form that exists only because the medium has no layering.
//
// They are computed rather than assumed because everything about the wave's
// excitation follows from them, and because they are independently checkable:
// the free-surface tractions must vanish, and the surface ellipticity must
// come out at the textbook 0.6813 for a Poisson solid. Those two checks
// between them pin the algebra.
type Eigenfunctions struct {
	// K is the Rayleigh wavenumber, omega / cR.
	K float64
	// Q and S are the P and SV vertical decay rates. Both are positive, so
	// both potentials decay into the half-space, which is what makes this a
	// surface wave.
	Q, S float64

	// coefficients of the two exponentials in each component
	xq, xs float64
	zq, zs float64
}

// NewEigenfunctions solves the Rayleigh mode shapes at angular frequency
// omega for the given half-space.
//
// The SV potential's amplitude is fixed relative to the P potential's by the
// vanishing of shear traction at the free surface, B = -2ikq/(k^2+s^2) * A.
// The remaining overall scale is arbitrary — it cancels in every quantity
// derived below, because each is a ratio of quadratics in the eigenfunctions.
func NewEigenfunctions(h soil.HalfSpace, omega float64) (Eigenfunctions, error) {
	cr, err := h.RayleighVelocity()
	if err != nil {
		return Eigenfunctions{}, err
	}
	if omega <= 0 {
		return Eigenfunctions{}, fmt.Errorf("green: angular frequency must be positive, got %g", omega)
	}
	c := float64(cr)
	k := omega / c
	// cR is below both body-wave speeds, so both radicands are positive and
	// both decay rates real. That is the definition of a trapped surface wave.
	q := k * math.Sqrt(1-(c*c)/(float64(h.Vp)*float64(h.Vp)))
	s := k * math.Sqrt(1-(c*c)/(float64(h.Vs)*float64(h.Vs)))

	k2, s2 := k*k, s*s
	e := Eigenfunctions{K: k, Q: q, S: s}
	// u_x(z) = i * [ k e^{-qz} - 2kqs/(k^2+s^2) e^{-sz} ]
	e.xq, e.xs = k, -2*k*q*s/(k2+s2)
	// u_z(z) =     [ -q e^{-qz} + 2k^2 q/(k^2+s^2) e^{-sz} ]
	e.zq, e.zs = -q, 2*k2*q/(k2+s2)
	return e, nil
}

// Horizontal is the radial displacement eigenfunction at depth z, up to the
// factor of i that puts it 90 degrees out of phase with the vertical. That
// phase difference is what makes Rayleigh particle motion elliptical.
func (e Eigenfunctions) Horizontal(z float64) float64 {
	return e.xq*math.Exp(-e.Q*z) + e.xs*math.Exp(-e.S*z)
}

// Vertical is the vertical displacement eigenfunction at depth z.
func (e Eigenfunctions) Vertical(z float64) float64 {
	return e.zq*math.Exp(-e.Q*z) + e.zs*math.Exp(-e.S*z)
}

// Ellipticity is the ratio of horizontal to vertical displacement amplitude
// at the free surface.
//
// It depends on Poisson's ratio alone — not on frequency, not on the medium's
// scale — and for a Poisson solid it is 0.6813. That makes it the sharpest
// available check that the eigenfunctions are right, since it is a pure number
// that falls out of the algebra and can be compared against a value derived
// independently.
func (e Eigenfunctions) Ellipticity() float64 {
	return math.Abs(e.Horizontal(0) / e.Vertical(0))
}

// EnergyIntegral is I1 = (1/2) * integral of rho * (ux^2 + uz^2) dz over the
// half-space, evaluated in closed form: every term is a product of decaying
// exponentials.
//
// It appears in the denominator of the excitation, where it normalises away
// the arbitrary overall amplitude of the mode shapes.
func (e Eigenfunctions) EnergyIntegral(density float64) float64 {
	// integral of (a e^{-qz} + b e^{-sz})^2 dz = a^2/2q + 2ab/(q+s) + b^2/2s
	sq := func(a, b float64) float64 {
		return a*a/(2*e.Q) + 2*a*b/(e.Q+e.S) + b*b/(2*e.S)
	}
	return 0.5 * density * (sq(e.xq, e.xs) + sq(e.zq, e.zs))
}

// HalfSpaceGF is the surface-to-surface vertical Green's function of a
// homogeneous half-space, in the far field.
type HalfSpaceGF struct {
	Soil soil.HalfSpace
	// RefFreq anchors the causal attenuation model: it is the frequency at
	// which the phase velocity equals the elastic Rayleigh velocity. Zero
	// defaults to 30 Hz, near the middle of the footstep band.
	RefFreq units.Hertz
}

const defaultRefFreq = 30.0

func (g HalfSpaceGF) refFreq() float64 {
	if g.RefFreq > 0 {
		return float64(g.RefFreq)
	}
	return defaultRefFreq
}

// gamma is Kjartansson's constant-Q exponent, (1/pi) * arctan(1/Q). It is
// small — about 0.013 for a Q of 25 — and it sets both how fast the medium
// attenuates and how much it disperses, which is the whole point: they are
// one parameter, not two.
func (g HalfSpaceGF) gamma() float64 {
	return math.Atan(1/g.Soil.Qs) / math.Pi
}

// ComplexWavenumber is k(omega) for the Rayleigh wave, carrying both the
// phase delay in its real part and the attenuation in its imaginary part.
//
// Attenuation and dispersion are not independent. A medium that absorbs cannot
// have a frequency-independent velocity without violating causality — the
// Kramers-Kronig relations tie the two together. Applying exp(-omega*r/2Qc) on
// its own, which is the obvious thing to do, produces energy arriving before
// the first arrival. That precursor is small, but it lands exactly where an
// arrival-time picker looks, so it would corrupt WP3's localisation while
// looking like a physical effect rather than a modelling error.
//
// This is Kjartansson's constant-Q model, k = (omega/c0) * (omega/omega0)^-g *
// exp(-i*pi*g/2), rather than the more commonly seen Futterman logarithmic
// pairing. Two reasons. Kjartansson is derived from a causal creep function
// and is therefore *exactly* causal, where Futterman is causal only to first
// order in 1/Q and leaves a percent-level precursor at the Q values soils
// actually have. And Futterman's logarithm diverges at zero frequency and
// needs an arbitrary low-frequency clamp, which breaks the very pairing it
// exists to provide; Kjartansson's power law does not.
//
// The two agree to better than a tenth of a percent for Q above about 20, so
// nothing is given up by preferring the one that is exact.
func (g HalfSpaceGF) ComplexWavenumber(f units.Hertz) (complex128, error) {
	cr, err := g.Soil.RayleighVelocity()
	if err != nil {
		return 0, err
	}
	if f <= 0 {
		return 0, nil
	}
	gam := g.gamma()
	f0 := g.refFreq()
	// c0 is scaled so the phase velocity comes out at exactly the elastic
	// Rayleigh velocity at the reference frequency.
	c0 := float64(cr) * math.Cos(math.Pi*gam/2)
	omega := 2 * math.Pi * float64(f)
	mag := omega / c0 * math.Pow(float64(f)/f0, -gam)
	return complex(mag, 0) * cmplx.Exp(complex(0, -math.Pi*gam/2)), nil
}

// PhaseVelocity is the Rayleigh phase velocity at f, which for a constant-Q
// medium is the power law cR * (f/f0)^gamma. Higher frequencies travel faster,
// by a fraction of a percent per decade at soil Q values.
func (g HalfSpaceGF) PhaseVelocity(f units.Hertz) (units.SpeedMPS, error) {
	cr, err := g.Soil.RayleighVelocity()
	if err != nil {
		return 0, err
	}
	if f <= 0 {
		return 0, nil
	}
	return units.SpeedMPS(float64(cr) * math.Pow(float64(f)/g.refFreq(), g.gamma())), nil
}

// VelocityResponse is the Green's function at range r and frequency f: the
// vertical ground *velocity* at the receiver per newton of vertical force at
// the source, in (m/s)/N.
//
// Velocity rather than displacement because that is what a geophone measures,
// so the conversion belongs here rather than being left for the sensor stage
// to remember.
//
// The magnitude is the standard single-mode surface-wave excitation,
//
//	uz(0)^2 / (8 * c * U * I1) * sqrt(2 / (pi * k * r))
//
// in which the arbitrary scale of the eigenfunctions cancels between the
// numerator and I1. The sqrt(1/r) is the geometric spreading of a wave
// confined to the surface, and it is the reason a footstep is detectable at
// range at all: body waves fall off as 1/r^2 along the free surface, so by
// 10 m essentially everything left is Rayleigh.
func (g HalfSpaceGF) VelocityResponse(r units.Metres, f units.Hertz) (complex128, error) {
	if r <= 0 {
		return 0, fmt.Errorf("green: range must be positive, got %g m", r)
	}
	if f <= 0 {
		return 0, nil // a velocity transducer sees nothing at DC anyway
	}
	c, err := g.PhaseVelocity(f)
	if err != nil {
		return 0, err
	}
	kc, err := g.ComplexWavenumber(f)
	if err != nil {
		return 0, err
	}
	omega := 2 * math.Pi * float64(f)
	e, err := NewEigenfunctions(g.Soil, omega)
	if err != nil {
		return 0, err
	}

	cc := float64(c)
	k := real(kc)
	// Non-dispersive to first order, so group velocity equals phase velocity;
	// the Q-induced dispersion is a fraction of a percent and enters the
	// amplitude only at second order.
	u := cc
	i1 := e.EnergyIntegral(float64(g.Soil.Density))
	uz := e.Vertical(0)

	amp := uz * uz / (8 * cc * u * i1) * math.Sqrt(2/(math.Pi*k*float64(r)))

	// Propagation. Both the phase delay and the attenuation live in the
	// complex wavenumber, so they cannot drift apart. The pi/4 is the
	// far-field asymptotic phase of the Hankel function — the signature of a
	// cylindrically spreading wave, not a free parameter.
	prop := cmplx.Exp(complex(0, math.Pi/4) - 1i*kc*complex(float64(r), 0))

	// Displacement to velocity is multiplication by i*omega.
	return complex(0, omega) * complex(amp, 0) * prop, nil
}

// Synthesise convolves a force time series with the Green's function at range
// r, returning ground velocity in m/s sampled at fs.
//
// It runs through ImpulseResponse rather than multiplying spectra directly,
// which is slower and is the right thing to do anyway. Multiplying the force
// spectrum by the frequency response gives a result that carries the acausal
// pre-ring of the band limit: a band-limited causal system is not causal, and
// the zero-phase truncation at Nyquist rings symmetrically in time. Measured
// here, that pre-ring reaches about 1% of peak in the low-amplitude coda —
// small, but it is energy arriving before it should, which is precisely the
// artefact this model works hardest to avoid.
//
// Taking the impulse response and keeping only its causal half removes it.
// Adding the discarded negative-time taps back reproduces the spectral-product
// answer to 3e-14, which is how the attribution was confirmed rather than
// assumed.
//
// The further benefit is that the offline and streaming paths now share one
// representation of the Green's function, so a trace synthesised in one piece
// and the same trace synthesised in chunks are the same trace, not two
// approximations that have to be reconciled.
func (g HalfSpaceGF) Synthesise(force []units.Newtons, fs float64, r units.Metres) ([]units.Velocity, error) {
	if err := g.Soil.Validate(); err != nil {
		return nil, err
	}
	if fs <= 0 {
		return nil, fmt.Errorf("green: sample rate must be positive, got %g", fs)
	}
	if r <= 0 {
		return nil, fmt.Errorf("green: range must be positive, got %g m", r)
	}
	if len(force) == 0 {
		return nil, nil
	}

	cr, err := g.Soil.RayleighVelocity()
	if err != nil {
		return nil, err
	}
	impulse, err := g.ImpulseResponse(fs, r, 0)
	if err != nil {
		return nil, err
	}

	raw := make([]float64, len(force))
	for i, v := range force {
		raw[i] = float64(v)
	}
	out := dsp.Convolve(raw, impulse)

	// Keep the source, the travel time, and a second of coda.
	keep := min(len(force)+int(math.Ceil(float64(r)/float64(cr)*fs))+int(fs), len(out))
	vel := make([]units.Velocity, keep)
	for i := range vel {
		vel[i] = units.Velocity(out[i])
	}
	return vel, nil
}

// TravelTime is when the Rayleigh arrival reaches range r, at the reference
// frequency.
func (g HalfSpaceGF) TravelTime(r units.Metres) (units.Seconds, error) {
	c, err := g.PhaseVelocity(units.Hertz(g.refFreq()))
	if err != nil {
		return 0, err
	}
	return units.Seconds(float64(r) / float64(c)), nil
}

// ImpulseResponse is the Green's function as a finite time-domain filter at
// sample rate fs: the ground velocity, sample by sample, that a unit impulse of
// force at the source produces at range r.
//
// The streaming synthesis in WP2 needs this form rather than the frequency
// response, because a chunk arriving every few milliseconds cannot be
// transformed against a spectrum defined over the whole record. Turning one
// into the other is a truncation, and truncation is a modelling choice: the
// coda of a Rayleigh arrival decays but never ends, so length trades fidelity
// against the state a real-time filter has to carry.
//
// The default length covers the travel time plus two seconds of coda, which
// puts the truncation about 80 dB below the peak for soils in the range this
// project cares about. Callers sweeping chunk size for O2 should hold this
// fixed, or they will be measuring two things at once.
func (g HalfSpaceGF) ImpulseResponse(fs float64, r units.Metres, length int) ([]float64, error) {
	if err := g.Soil.Validate(); err != nil {
		return nil, err
	}
	if fs <= 0 {
		return nil, fmt.Errorf("green: sample rate must be positive, got %g", fs)
	}
	if r <= 0 {
		return nil, fmt.Errorf("green: range must be positive, got %g m", r)
	}
	if length <= 0 {
		cr, err := g.Soil.RayleighVelocity()
		if err != nil {
			return nil, err
		}
		length = int(math.Ceil((float64(r)/float64(cr) + 2) * fs))
	}

	return dsp.CausalImpulseResponse(length, fs, func(f float64) (complex128, error) {
		return g.VelocityResponse(r, units.Hertz(f))
	})
}
