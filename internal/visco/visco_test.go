package visco

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestFitHitsTheTargetExactly(t *testing.T) {
	for _, q := range []float64{5, 25, 100, 1000} {
		for _, f0 := range []float64{1, 30, 250} {
			s, err := Fit(q, f0)
			if err != nil {
				t.Fatal(err)
			}
			if got := s.Q(f0); math.Abs(got-q) > 1e-12*q {
				t.Errorf("Q(%g Hz) = %.12g, want %g", f0, got, q)
			}
		}
	}
}

// The fitted frequency must be the minimum of Q, not merely a point on the
// curve. Fit inverts a stationarity condition, so if the inversion were wrong
// the target value could still be hit at a frequency that is not the minimum.
func TestFittedFrequencyIsTheMinimum(t *testing.T) {
	s, err := Fit(25, 30)
	if err != nil {
		t.Fatal(err)
	}
	base := s.Q(30)
	for _, f := range []float64{1, 10, 20, 29, 31, 45, 90, 900} {
		if s.Q(f) < base {
			t.Errorf("Q(%g Hz) = %g is below the fitted minimum %g", f, s.Q(f), base)
		}
	}
	// And it is a genuine minimum, not a flat spot.
	if s.Q(3) < 2*base {
		t.Errorf("Q(3 Hz) = %g, expected a single mechanism to rise well away from f0", s.Q(3))
	}
}

func TestModulusLimits(t *testing.T) {
	s, _ := Fit(25, 30)
	if got := s.Modulus(0); got != 1 {
		t.Errorf("R(0) = %v, want 1", got)
	}
	// R(inf) = 1 + Delta, approached from below.
	if got, want := real(s.Modulus(1e12)), 1+s.Delta(); math.Abs(got-want) > 1e-6*want {
		t.Errorf("R(inf) = %g, want %g", got, want)
	}
	if s.Delta() <= 0 {
		t.Errorf("Delta = %g, want positive for a dissipative solid", s.Delta())
	}
}

// The whole point of RelaxedVelocity: the nominal velocity is the phase
// velocity at the reference frequency, so a layer's Vs keeps its elastic
// meaning.
func TestPhaseVelocityIsAnchoredAtTheReference(t *testing.T) {
	s, _ := Fit(25, 30)
	for _, v := range []float64{120, 200, 800} {
		if got := s.PhaseVelocity(v, 30, 30); math.Abs(got-v) > 1e-10*v {
			t.Errorf("phase velocity at fRef = %.12g, want %g", got, v)
		}
	}
}

// Causal dissipation means velocity rises with frequency. A model that lost
// energy without dispersing would violate Kramers-Kronig, and is the classic
// way an attenuation model goes acausal.
func TestVelocityDispersionIsMonotonic(t *testing.T) {
	s, _ := Fit(25, 30)
	prev := 0.0
	for f := 0.5; f < 500; f *= 1.2 {
		v := s.PhaseVelocity(200, 30, f)
		if v <= prev {
			t.Fatalf("phase velocity fell at %g Hz: %g after %g", f, v, prev)
		}
		prev = v
	}
	// The total rise across the band is the 1/Q-order effect it should be.
	lo, hi := s.PhaseVelocity(200, 30, 0.5), s.PhaseVelocity(200, 30, 500)
	if rise := hi/lo - 1; rise < 0.01 || rise > 0.15 {
		t.Errorf("velocity rise over the band = %.3f, want an order-1/Q effect", rise)
	}
}

// The complex velocity must have the same sign convention as the Kjartansson
// model it stands in for, or waves would grow with range instead of decaying.
func TestAttenuationHasTheRightSign(t *testing.T) {
	s, _ := Fit(25, 30)
	c := s.Velocity(200, 30, 60)
	if imag(c) <= 0 {
		t.Fatalf("Im(c) = %g, want positive so that Im(omega/c) is negative", imag(c))
	}
	// Over one wavelength the amplitude should fall by exp(-pi/Q).
	w := 2 * math.Pi * 60
	k := complex(w, 0) / c
	lambda := 2 * math.Pi / real(k)
	decay := math.Exp(imag(k) * lambda)
	if want := math.Exp(-math.Pi / s.Q(60)); math.Abs(decay-want) > 0.01*want {
		t.Errorf("decay per wavelength = %.4f, want exp(-pi/Q) = %.4f", decay, want)
	}
}

func TestElasticLimit(t *testing.T) {
	s := SLS{TauSigma: 1e-3, TauEps: 1e-3}
	if got := s.Modulus(50); got != 1 {
		t.Errorf("R = %v, want exactly 1 when the relaxation times coincide", got)
	}
	if !math.IsInf(s.Q(50), 1) {
		t.Errorf("Q = %g, want infinite", s.Q(50))
	}
	if got := s.PhaseVelocity(200, 30, 500); math.Abs(got-200) > 1e-12 {
		t.Errorf("phase velocity = %g, want 200 with no dissipation", got)
	}
}

// Moduli must be the same numbers the frequency-domain path uses, or the two
// sides of the slice 4 comparison would be modelling different media. The
// identity is M(f) = M_R * R(f) = rho * c(f)^2.
func TestModuliAgreeWithTheComplexVelocity(t *testing.T) {
	s, _ := Fit(25, 30)
	const rho, v, fRef = 1700, 200, 30
	mr, mu := s.Moduli(v, fRef, rho)
	if want := mr * (1 + s.Delta()); math.Abs(mu-want) > 1e-9*want {
		t.Errorf("unrelaxed modulus = %g, want %g", mu, want)
	}
	for _, f := range []float64{1, 30, 200} {
		c := s.Velocity(v, fRef, f)
		got := complex(rho, 0) * c * c
		want := complex(mr, 0) * s.Modulus(f)
		if cmplx.Abs(got-want) > 1e-9*cmplx.Abs(want) {
			t.Errorf("at %g Hz: rho*c^2 = %v, M_R*R = %v", f, got, want)
		}
	}
}

func TestFitRejectsNonsense(t *testing.T) {
	if _, err := Fit(0, 30); err == nil {
		t.Error("expected an error for Q = 0")
	}
	if _, err := Fit(25, 0); err == nil {
		t.Error("expected an error for f0 = 0")
	}
}
