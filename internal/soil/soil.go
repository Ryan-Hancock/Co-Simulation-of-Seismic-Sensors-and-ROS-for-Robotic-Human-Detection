// Package soil describes the elastic medium a footstep radiates into.
//
// Slice 0 needs only a homogeneous half-space: one set of elastic constants,
// no layering, no depth dependence. That is deliberately less than the ground
// under a real footstep, and the plan says so — layering, dispersion and the
// frequency-dependence they bring arrive in slice 3. What is here is exact for
// what it claims, and it carries the Rayleigh root solver that the layered
// model will generalise rather than replace.
package soil

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/units"
)

// HalfSpace is a homogeneous, isotropic, linearly elastic half-space with
// anelastic attenuation.
//
// Parameterised by wave speeds and density rather than by Lame constants:
// that is how field measurements arrive (a seismic refraction survey reports
// velocities) and how the domain-randomisation axes in O4 are naturally
// expressed.
type HalfSpace struct {
	// Vp is the compressional wave speed.
	Vp units.SpeedMPS
	// Vs is the shear wave speed.
	Vs units.SpeedMPS
	// Density is the bulk density.
	Density units.DensityKgM3
	// Qs is the shear quality factor: the reciprocal of the fraction of
	// energy lost per radian of oscillation. Soils are lossy, with Qs in the
	// tens, and at footstep frequencies and ranges this is not a small
	// correction — it is much of why a footstep is undetectable at 50 m.
	Qs float64
	// Qp is the compressional quality factor. Zero defaults to 9/4 * Qs,
	// the value implied when loss is entirely in shear.
	Qp float64
}

// Named soils spanning the range this project cares about. These are
// plausible mid-points, not measurements: WP4 replaces them with surveyed
// values, and O4 randomises around them. They exist so that a config file can
// say "dry sand" and mean something reproducible.
func DrySand() HalfSpace {
	return HalfSpace{Vp: 350, Vs: 150, Density: 1600, Qs: 20}
}

func Loam() HalfSpace {
	return HalfSpace{Vp: 500, Vs: 200, Density: 1700, Qs: 25}
}

func FirmSoil() HalfSpace {
	return HalfSpace{Vp: 750, Vs: 300, Density: 1900, Qs: 40}
}

func WeatheredRock() HalfSpace {
	return HalfSpace{Vp: 1800, Vs: 800, Density: 2300, Qs: 80}
}

// PoissonSolid is the classical test medium, Vp = sqrt(3)*Vs, for which the
// Rayleigh velocity has an exact closed form. It is a fixture, not a soil.
func PoissonSolid(vs units.SpeedMPS, density units.DensityKgM3) HalfSpace {
	return HalfSpace{Vp: units.SpeedMPS(math.Sqrt(3) * float64(vs)), Vs: vs, Density: density, Qs: 1e9}
}

// Validate reports whether the constants describe a physically possible
// medium. The binding constraint is Vp > sqrt(2)*Vs: below it the Poisson
// ratio goes negative, which real soils do not do and which would send the
// Rayleigh root solver looking for a root that is not there.
func (h HalfSpace) Validate() error {
	switch {
	case h.Vs <= 0:
		return fmt.Errorf("soil: Vs must be positive, got %g m/s", h.Vs)
	case h.Vp <= 0:
		return fmt.Errorf("soil: Vp must be positive, got %g m/s", h.Vp)
	case h.Density <= 0:
		return fmt.Errorf("soil: density must be positive, got %g kg/m^3", h.Density)
	case float64(h.Vp) <= math.Sqrt2*float64(h.Vs):
		return fmt.Errorf("soil: Vp=%g m/s must exceed sqrt(2)*Vs=%g m/s for a non-negative Poisson ratio",
			h.Vp, math.Sqrt2*float64(h.Vs))
	case h.Qs <= 0:
		return fmt.Errorf("soil: Qs must be positive, got %g", h.Qs)
	case h.Qp < 0:
		return fmt.Errorf("soil: Qp must not be negative, got %g", h.Qp)
	}
	return nil
}

// ShearModulus is mu = rho * Vs^2.
func (h HalfSpace) ShearModulus() units.Pascals {
	return units.Pascals(float64(h.Density) * float64(h.Vs) * float64(h.Vs))
}

// LameLambda is lambda = rho * (Vp^2 - 2*Vs^2).
func (h HalfSpace) LameLambda() units.Pascals {
	vp, vs := float64(h.Vp), float64(h.Vs)
	return units.Pascals(float64(h.Density) * (vp*vp - 2*vs*vs))
}

// PoissonRatio is nu = (Vp^2 - 2 Vs^2) / (2 (Vp^2 - Vs^2)).
func (h HalfSpace) PoissonRatio() float64 {
	vp2, vs2 := float64(h.Vp)*float64(h.Vp), float64(h.Vs)*float64(h.Vs)
	return (vp2 - 2*vs2) / (2 * (vp2 - vs2))
}

// QsOrQp returns the compressional quality factor, defaulting to 9/4 * Qs.
//
// That ratio is what follows from putting all the loss in shear and none in
// bulk, which is the usual assumption for soils and the usual default in the
// literature.
func (h HalfSpace) QpEffective() float64 {
	if h.Qp > 0 {
		return h.Qp
	}
	return 2.25 * h.Qs
}

// RayleighVelocity is the speed of the Rayleigh surface wave.
//
// It is the root in (0, Vs) of the secular equation
//
//	(2 - x)^2 = 4 * sqrt(1-x) * sqrt(1 - x*(Vs/Vp)^2),   x = (c/Vs)^2
//
// solved by bisection on the irrational form rather than on the rationalised
// cubic. The cubic is tempting — it is a closed form — but squaring introduces
// two spurious roots that are not Rayleigh waves, and picking the right one
// among three means knowing the answer already. Bisecting the original has a
// single crossing to find, and it is the same routine the layered solver in
// slice 3 will run against a much less well-behaved secular function.
//
// Rayleigh speed is always between 0.874*Vs (Poisson ratio 0) and 0.955*Vs
// (0.5), so the answer is never surprising; the point of computing it is that
// the weak dependence on Poisson ratio is real and worth carrying.
func (h HalfSpace) RayleighVelocity() (units.SpeedMPS, error) {
	if err := h.Validate(); err != nil {
		return 0, err
	}
	k := float64(h.Vs) / float64(h.Vp)
	k2 := k * k

	// F(0) = 0 is a root of the algebra but not a wave. F is negative just
	// above it and positive at x=1, so the bracket starts clear of the origin.
	f := func(x float64) float64 {
		return (2-x)*(2-x) - 4*math.Sqrt(1-x)*math.Sqrt(1-x*k2)
	}
	lo, hi := 1e-9, 1.0
	if f(lo) >= 0 || f(hi) <= 0 {
		return 0, fmt.Errorf("soil: no Rayleigh root bracketed for Vp=%g Vs=%g", h.Vp, h.Vs)
	}
	for range 200 {
		mid := 0.5 * (lo + hi)
		if mid == lo || mid == hi {
			break // converged to adjacent floats
		}
		if f(mid) < 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	return units.SpeedMPS(math.Sqrt(0.5*(lo+hi)) * float64(h.Vs)), nil
}

// String renders the medium the way a log line wants it.
func (h HalfSpace) String() string {
	return fmt.Sprintf("Vp=%.0f Vs=%.0f rho=%.0f Qs=%.0f nu=%.3f", h.Vp, h.Vs, h.Density, h.Qs, h.PoissonRatio())
}
