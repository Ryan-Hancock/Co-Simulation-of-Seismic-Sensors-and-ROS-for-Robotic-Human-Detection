package soil

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/units"
)

// V1: the Rayleigh velocity of a Poisson solid, against its exact closed form.
//
// For nu = 1/4 the rationalised secular cubic factors as
//
//	3x^3 - 24x^2 + 56x - 32 = (x - 4)(3x^2 - 12x + 8)
//
// whose physical root is x = 2 - 2/sqrt(3), so cR/Vs = sqrt(2 - 2/sqrt(3)) =
// 0.9194... . That is an exact target, not a rounded literal from a textbook,
// which is what makes it worth testing against: the tolerance can be set at
// the solver's convergence rather than at the precision of a printed constant.
func TestRayleighVelocityPoissonSolid(t *testing.T) {
	h := PoissonSolid(200, 1900)

	if nu := h.PoissonRatio(); math.Abs(nu-0.25) > 1e-12 {
		t.Fatalf("fixture is not a Poisson solid: nu = %g", nu)
	}

	want := math.Sqrt(2-2/math.Sqrt(3)) * float64(h.Vs)
	got, err := h.RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}
	if rel := math.Abs(float64(got)-want) / want; rel > 1e-14 {
		t.Errorf("cR = %.12f m/s, want %.12f (rel err %g)", got, want, rel)
	}
	// And the ratio itself, the number every textbook prints.
	if ratio := float64(got) / float64(h.Vs); math.Abs(ratio-0.9194) > 1e-4 {
		t.Errorf("cR/Vs = %.6f, want 0.9194", ratio)
	}
}

// The Rayleigh ratio depends only on Poisson's ratio. Rather than compare
// against literals transcribed from a textbook, this checks the solver against
// the *rationalised* secular cubic
//
//	x^3 - 8x^2 + 8x(3 - 2k^2) - 16(1 - k^2) = 0,	k = Vs/Vp
//
// solved independently here. Squaring the secular equation to reach that cubic
// introduces two spurious roots, which is exactly why the solver does not use
// it — but as a check it is an algebraically different route to the same
// number, and scanning for the root in (0,1) sidesteps the root-selection
// problem that makes the cubic unattractive in production.
func TestRayleighAgainstRationalisedCubic(t *testing.T) {
	// Root of the cubic in (0,1), found by scanning for the sign change so no
	// prior knowledge of which root is physical is needed.
	cubicRoot := func(k2 float64) float64 {
		f := func(x float64) float64 {
			return x*x*x - 8*x*x + 8*x*(3-2*k2) - 16*(1-k2)
		}
		lo, hi := 1e-9, 1.0
		if f(lo)*f(hi) >= 0 {
			t.Fatalf("cubic has no sign change in (0,1) for k2=%g", k2)
		}
		for range 200 {
			mid := 0.5 * (lo + hi)
			if mid == lo || mid == hi {
				break
			}
			if f(lo)*f(mid) <= 0 {
				hi = mid
			} else {
				lo = mid
			}
		}
		return 0.5 * (lo + hi)
	}

	for vpvs := 1.45; vpvs < 8; vpvs += 0.05 {
		h := HalfSpace{Vp: units.SpeedMPS(200 * vpvs), Vs: 200, Density: 1800, Qs: 30}
		cr, err := h.RayleighVelocity()
		if err != nil {
			t.Fatalf("Vp/Vs=%g: %v", vpvs, err)
		}
		k := 1 / vpvs
		want := math.Sqrt(cubicRoot(k*k)) * float64(h.Vs)
		if rel := math.Abs(float64(cr)-want) / want; rel > 1e-12 {
			t.Errorf("Vp/Vs=%.2f (nu=%.3f): cR = %.9f, cubic gives %.9f (rel err %g)",
				vpvs, h.PoissonRatio(), cr, want, rel)
		}
	}
}

func TestRayleighRatioIsMonotonicInPoissonRatio(t *testing.T) {
	var prevNu, prevRatio float64 = -1, -1
	for vpvs := 1.45; vpvs < 6; vpvs += 0.05 {
		h := HalfSpace{Vp: units.SpeedMPS(200 * vpvs), Vs: 200, Density: 1800, Qs: 30}
		cr, err := h.RayleighVelocity()
		if err != nil {
			t.Fatalf("Vp/Vs=%g: %v", vpvs, err)
		}
		nu, ratio := h.PoissonRatio(), float64(cr)/float64(h.Vs)
		if ratio < 0.874 || ratio > 0.9554 {
			t.Errorf("Vp/Vs=%g: cR/Vs = %g outside the physical band [0.874, 0.955]", vpvs, ratio)
		}
		if prevRatio > 0 && ratio <= prevRatio {
			t.Errorf("cR/Vs not increasing with nu: %g at nu=%g, then %g at nu=%g",
				prevRatio, prevNu, ratio, nu)
		}
		prevNu, prevRatio = nu, ratio
	}
}

// The Rayleigh wave is always slower than shear, which is always slower than
// compressional. If this ordering ever breaks the medium is unphysical, and
// every travel-time in the model above it is wrong.
func TestVelocityOrderingForNamedSoils(t *testing.T) {
	for name, h := range map[string]HalfSpace{
		"dry sand": DrySand(), "loam": Loam(),
		"firm soil": FirmSoil(), "weathered rock": WeatheredRock(),
	} {
		t.Run(name, func(t *testing.T) {
			if err := h.Validate(); err != nil {
				t.Fatalf("preset is not physical: %v", err)
			}
			cr, err := h.RayleighVelocity()
			if err != nil {
				t.Fatal(err)
			}
			if !(cr < h.Vs && h.Vs < h.Vp) {
				t.Errorf("expected cR < Vs < Vp, got cR=%g Vs=%g Vp=%g", cr, h.Vs, h.Vp)
			}
			if nu := h.PoissonRatio(); nu <= 0 || nu >= 0.5 {
				t.Errorf("Poisson ratio %g outside (0, 0.5)", nu)
			}
		})
	}
}

// Elastic moduli against their definitions, since everything downstream scales
// by the shear modulus and an error there is a pure amplitude offset — the
// kind that looks like a calibration problem rather than a bug.
func TestElasticModuli(t *testing.T) {
	h := HalfSpace{Vp: 500, Vs: 200, Density: 1700, Qs: 25}
	if want := 1700.0 * 200 * 200; math.Abs(float64(h.ShearModulus())-want) > 1e-6 {
		t.Errorf("mu = %g, want %g", h.ShearModulus(), want)
	}
	if want := 1700.0 * (500*500 - 2*200*200); math.Abs(float64(h.LameLambda())-want) > 1e-6 {
		t.Errorf("lambda = %g, want %g", h.LameLambda(), want)
	}
	// Recovering Vp and Vs from the moduli closes the loop.
	mu, lam, rho := float64(h.ShearModulus()), float64(h.LameLambda()), float64(h.Density)
	if vs := math.Sqrt(mu / rho); math.Abs(vs-float64(h.Vs)) > 1e-9 {
		t.Errorf("Vs recovered from mu = %g, want %g", vs, h.Vs)
	}
	if vp := math.Sqrt((lam + 2*mu) / rho); math.Abs(vp-float64(h.Vp)) > 1e-9 {
		t.Errorf("Vp recovered from moduli = %g, want %g", vp, h.Vp)
	}
}

func TestQpDefaultsFromQs(t *testing.T) {
	h := HalfSpace{Vp: 500, Vs: 200, Density: 1700, Qs: 20}
	if got := h.QpEffective(); math.Abs(got-45) > 1e-12 {
		t.Errorf("Qp = %g, want 45 (9/4 * Qs)", got)
	}
	h.Qp = 100
	if got := h.QpEffective(); got != 100 {
		t.Errorf("explicit Qp = %g, want 100", got)
	}
}

func TestValidateCatchesUnphysicalMedia(t *testing.T) {
	for name, h := range map[string]HalfSpace{
		"zero Vs":         {Vp: 500, Vs: 0, Density: 1700, Qs: 25},
		"zero Vp":         {Vp: 0, Vs: 200, Density: 1700, Qs: 25},
		"zero density":    {Vp: 500, Vs: 200, Density: 0, Qs: 25},
		"negative Q":      {Vp: 500, Vs: 200, Density: 1700, Qs: -1},
		"Vp below sqrt2":  {Vp: 250, Vs: 200, Density: 1700, Qs: 25},
		"Vp equals sqrt2": {Vp: units.SpeedMPS(math.Sqrt2 * 200), Vs: 200, Density: 1700, Qs: 25},
	} {
		t.Run(name, func(t *testing.T) {
			if err := h.Validate(); err == nil {
				t.Errorf("Validate accepted %s: %s", name, h)
			}
			if _, err := h.RayleighVelocity(); err == nil {
				t.Errorf("RayleighVelocity accepted %s", name)
			}
		})
	}
}

func BenchmarkRayleighVelocity(b *testing.B) {
	h := Loam()
	for b.Loop() {
		if _, err := h.RayleighVelocity(); err != nil {
			b.Fatal(err)
		}
	}
}
