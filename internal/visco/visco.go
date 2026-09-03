// Package visco is the one attenuation model that both propagation paths can
// implement exactly.
//
// The f-k path (internal/fk) uses Kjartansson's constant-Q law, which is the
// right production model: exactly causal, no arbitrary low-frequency clamp,
// and constant Q is what soils actually do. But it has no finite-difference
// counterpart. A time-domain scheme cannot evaluate a fractional power of
// frequency; it needs a relaxation spectrum, and matching Kjartansson's over a
// band takes several fitted mechanisms with a fit error of its own.
//
// That fit error would land squarely in the middle of the slice 4 comparison.
// V5 asks whether the layered f-k solve is right, and the answer must not be
// clouded by a percent-level disagreement in the material model that is
// present by construction. So the comparison runs in a medium both sides
// represent *exactly*: a single standard linear solid.
//
// A single SLS has a Q that varies with frequency — a minimum at one frequency
// rising either side — where soils have Q roughly constant. That makes this a
// comparison medium, not a soil. It is deliberately not what the production
// path uses; it exists so that any difference the comparison finds is
// propagation, not parameterisation.
//
// The relaxation is scalar: one R(omega) multiplying the whole modulus tensor,
// so lambda and mu relax together and Qp equals Qs. Real rock puts most loss
// in shear (Qp = 9/4 Qs, which is what internal/fk assumes). Independent
// relaxation of the two moduli needs a second memory variable per stress
// component and buys nothing for a comparison, since both sides would still
// agree.
package visco

import (
	"fmt"
	"math"
	"math/cmplx"
)

// SLS is a standard linear solid, given by its two relaxation times.
//
// The stress relaxation function is psi(t) = M_R*(1 + Delta*exp(-t/TauSigma)),
// whose complex modulus is M_R*(1 + i*w*TauEps)/(1 + i*w*TauSigma). TauEps
// exceeds TauSigma for a dissipative solid; equal times are the elastic limit.
type SLS struct {
	// TauSigma is the stress relaxation time, in seconds.
	TauSigma float64
	// TauEps is the strain relaxation time, in seconds.
	TauEps float64
}

// Fit returns the solid whose quality factor reaches its minimum q at f0.
//
// Q(w) = (1 + w^2*Te*Ts)/(w*(Te - Ts)) is minimised at w0 = 1/sqrt(Te*Ts),
// where it takes the value 2*sqrt(Te*Ts)/(Te - Ts). Inverting those two
// relations for Te and Ts is exact, so the fitted solid hits the target Q at
// the target frequency to machine precision rather than approximately.
func Fit(q, f0 float64) (SLS, error) {
	if q <= 0 {
		return SLS{}, fmt.Errorf("visco: Q must be positive, got %g", q)
	}
	if f0 <= 0 {
		return SLS{}, fmt.Errorf("visco: reference frequency must be positive, got %g Hz", f0)
	}
	w0 := 2 * math.Pi * f0
	root := math.Sqrt(1 + 1/(q*q))
	return SLS{
		TauSigma: (root - 1/q) / w0,
		TauEps:   (root + 1/q) / w0,
	}, nil
}

// Delta is the relaxation strength, TauEps/TauSigma - 1.
//
// It is the fractional gap between the unrelaxed modulus, which governs the
// instantaneous response a time-domain scheme applies first, and the relaxed
// one the memory variable pulls back towards.
func (s SLS) Delta() float64 { return s.TauEps/s.TauSigma - 1 }

// Modulus is R(f): the dimensionless modulus multiplier, R(0) = 1.
func (s SLS) Modulus(f float64) complex128 {
	w := complex(0, 2*math.Pi*f)
	return (1 + w*complex(s.TauEps, 0)) / (1 + w*complex(s.TauSigma, 0))
}

// Q at a frequency. Infinite at zero frequency and in the elastic limit.
func (s SLS) Q(f float64) float64 {
	m := s.Modulus(f)
	if imag(m) == 0 {
		return math.Inf(1)
	}
	return real(m) / imag(m)
}

// RelaxedVelocity is the velocity constant c0 for which the *phase* velocity
// equals v at fRef.
//
// The distinction matters and is easy to get wrong. A viscoelastic medium has
// no single velocity: the complex velocity is c0*sqrt(R(f)), and the speed a
// wavefront actually travels at is 1/Re(1/c), not |c| and not c0. Anchoring
// the phase velocity at a stated frequency is what lets a layer.Layer's Vs
// keep meaning the same thing it means in the elastic solver.
func (s SLS) RelaxedVelocity(v, fRef float64) float64 {
	return v * real(1/cmplx.Sqrt(s.Modulus(fRef)))
}

// Velocity is the complex velocity at f for a medium whose phase velocity is v
// at fRef.
func (s SLS) Velocity(v, fRef, f float64) complex128 {
	return complex(s.RelaxedVelocity(v, fRef), 0) * cmplx.Sqrt(s.Modulus(f))
}

// PhaseVelocity at f for a medium whose phase velocity is v at fRef.
func (s SLS) PhaseVelocity(v, fRef, f float64) float64 {
	return 1 / real(1/s.Velocity(v, fRef, f))
}

// Moduli returns the relaxed and unrelaxed real moduli for a wave whose phase
// velocity is v at fRef in a medium of the given density.
//
// These are what a finite-difference scheme needs: it steps with the unrelaxed
// modulus and lets a memory variable relax the stress towards M_R.
func (s SLS) Moduli(v, fRef, density float64) (relaxed, unrelaxed float64) {
	c0 := s.RelaxedVelocity(v, fRef)
	relaxed = density * c0 * c0
	return relaxed, relaxed * (1 + s.Delta())
}
