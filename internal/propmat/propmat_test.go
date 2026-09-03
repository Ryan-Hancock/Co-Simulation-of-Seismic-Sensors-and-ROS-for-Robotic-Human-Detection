package propmat

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
)

func testLayer() layer.Layer {
	return layer.Layer{Thickness: 3, Vp: 400, Vs: 160, Density: 1600}
}

// The propagator is the exponential of a traceless matrix, so its determinant
// is exactly one, and it must satisfy the semigroup property. Neither depends
// on any seismological reference being correct — they follow from what a
// propagator is — which makes them the sharpest available check that the
// system matrix and its exponential are right.
//
// They are also what caught the first version of this package. Written in
// physical units the motion-stress vector mixes metres with pascals, so the
// system matrix spanned sixteen orders of magnitude; det came out at 1.003 and
// P(d1)P(d2) differed from P(d1+d2) by a factor of two. Nothing downstream
// could have survived that, and no dispersion curve at low frequency showed
// it, because a thin layer's propagator is near the identity and hides its own
// conditioning.
func TestPropagatorIdentities(t *testing.T) {
	l := testLayer()
	muRef := l.ShearModulus()
	omega := 2 * math.Pi * 55.93
	k := omega / 150

	a := SystemMatrix(l, k, omega, muRef)
	expZeta := func(zeta float64) *mat.Dense {
		s := mat.NewDense(4, 4, nil)
		s.Scale(zeta, a)
		p := mat.NewDense(4, 4, nil)
		p.Exp(s)
		return p
	}

	t.Run("system matrix is traceless", func(t *testing.T) {
		var tr float64
		for i := range 4 {
			tr += a.At(i, i)
		}
		if math.Abs(tr) > 1e-14 {
			t.Errorf("trace = %g, want 0", tr)
		}
	})

	t.Run("determinant is one", func(t *testing.T) {
		// Exactly one in theory. Numerically the error grows with the norm of
		// A*zeta, because scaling-and-squaring loses a little accuracy for
		// every doubling — which is the reason LayerPropagator factors the
		// growth out before exponentiating rather than after. The tolerance
		// tracks that rather than pretending it is not there.
		for _, c := range []struct {
			zeta, tol float64
		}{{0.1, 1e-13}, {1, 1e-12}, {3, 1e-10}, {10, 1e-6}} {
			if d := mat.Det(expZeta(c.zeta)); math.Abs(d-1) > c.tol {
				t.Errorf("det P(%g) = %.12f, want 1 to within %g", c.zeta, d, c.tol)
			}
		}
	})

	t.Run("semigroup", func(t *testing.T) {
		p1, p2, p3 := expZeta(1.2), expZeta(1.8), expZeta(3.0)
		var prod mat.Dense
		prod.Mul(p1, p2)
		for i := range 4 {
			for j := range 4 {
				got, want := prod.At(i, j), p3.At(i, j)
				if math.Abs(got-want) > 1e-10*(1+math.Abs(want)) {
					t.Fatalf("P(1.2)P(1.8)[%d][%d] = %g, want P(3.0) = %g", i, j, got, want)
				}
			}
		}
	})

	t.Run("well conditioned", func(t *testing.T) {
		// Every entry of the dimensionless system matrix should be of order
		// one. If this ever fails the scaling has been undone and the
		// identities above will start to drift.
		for i := range 4 {
			for j := range 4 {
				if v := math.Abs(a.At(i, j)); v > 1e3 {
					t.Errorf("A[%d][%d] = %g; the system matrix is no longer scaled", i, j, v)
				}
			}
		}
	})
}

// The compound matrix must be a homomorphism: taking minors commutes with
// multiplying propagators. This is what makes the minor recursion solve the
// same problem as the direct one, and a wrong index ordering in Compound
// breaks it while leaving something that still looks like a dispersion
// function.
func TestCompoundIsHomomorphic(t *testing.T) {
	l := testLayer()
	muRef := l.ShearModulus()
	omega := 2 * math.Pi * 55.93
	k := omega / 150
	a := SystemMatrix(l, k, omega, muRef)

	expZeta := func(zeta float64) *mat.Dense {
		s := mat.NewDense(4, 4, nil)
		s.Scale(zeta, a)
		p := mat.NewDense(4, 4, nil)
		p.Exp(s)
		return p
	}
	p1, p2, p3 := expZeta(1.2), expZeta(1.8), expZeta(3.0)

	var want mat.Dense
	want.Mul(Compound(p1), Compound(p2))
	got := Compound(p3)
	for i := range 6 {
		for j := range 6 {
			if math.Abs(want.At(i, j)-got.At(i, j)) > 1e-9*(1+math.Abs(got.At(i, j))) {
				t.Fatalf("C(P1)C(P2)[%d][%d] = %g, want C(P1 P2) = %g", i, j, want.At(i, j), got.At(i, j))
			}
		}
	}
}

// The generalised Laplace expansion the secular function uses, checked against
// a direct determinant on an arbitrary matrix. Its signs are easy to get wrong
// and the error would be invisible: a sign flip turns the secular function
// into a different smooth function with different zeros.
func TestLaplaceExpansionSigns(t *testing.T) {
	r := mat.NewDense(4, 4, []float64{
		0.3, -1.2, 0.7, 2.1,
		1.1, 0.4, -0.9, 0.5,
		-0.6, 2.2, 1.3, -1.7,
		0.8, -0.3, 0.2, 1.9,
	})
	var mB, mE [6]float64
	for n, ij := range pairs {
		i, j := ij[0], ij[1]
		mB[n] = r.At(i, 0)*r.At(j, 1) - r.At(j, 0)*r.At(i, 1)
		mE[n] = r.At(i, 2)*r.At(j, 3) - r.At(j, 2)*r.At(i, 3)
	}
	var got float64
	for n := range 6 {
		got += laplaceSign[n] * mB[n] * mE[comp[n]]
	}
	if want := mat.Det(r); math.Abs(got-want) > 1e-12 {
		t.Errorf("Laplace expansion %.12f, want det %.12f", got, want)
	}
}

// With no layers the secular function must reduce to the classical Rayleigh
// equation, whose root slice 0 computes by a completely different route. This
// is the check that the half-space boundary condition, the vector convention
// and the scaling all line up before any layering is trusted.
func TestHomogeneousReducesToTheRayleighRoot(t *testing.T) {
	for name, h := range map[string]soil.HalfSpace{
		"Poisson solid":  soil.PoissonSolid(200, 1900),
		"dry sand":       soil.DrySand(),
		"loam":           soil.Loam(),
		"weathered rock": soil.WeatheredRock(),
	} {
		t.Run(name, func(t *testing.T) {
			want, err := h.RayleighVelocity()
			if err != nil {
				t.Fatal(err)
			}
			st := layer.Uniform(h)
			// Bisect the secular function directly.
			lo, hi := 0.8*float64(h.Vs), float64(h.Vs)*(1-1e-12)
			flo, err := SecularAtVelocity(st, 20, lo)
			if err != nil {
				t.Fatal(err)
			}
			for range 200 {
				mid := 0.5 * (lo + hi)
				v, err := SecularAtVelocity(st, 20, mid)
				if err != nil {
					t.Fatal(err)
				}
				if (flo < 0) != (v < 0) {
					hi = mid
				} else {
					lo, flo = mid, v
				}
			}
			got := 0.5 * (lo + hi)
			if rel := math.Abs(got-float64(want)) / float64(want); rel > 1e-9 {
				t.Errorf("layered solver gives cR = %.6f, closed form gives %.6f (rel %g)", got, want, rel)
			}
		})
	}
}

// A homogeneous half-space does not disperse, so the secular function must not
// depend on frequency at fixed phase velocity. Any drift means frequency is
// leaking in somewhere it should have cancelled.
func TestHomogeneousDoesNotDisperse(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	var first float64
	for i, f := range []float64{1, 5, 20, 100, 500} {
		v, err := SecularAtVelocity(st, f, 180)
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			first = v
			continue
		}
		if rel := math.Abs(v-first) / math.Abs(first); rel > 1e-12 {
			t.Errorf("secular function at 180 m/s moved by %g between 1 Hz and %g Hz", rel, f)
		}
	}
}

// V4: the reason for the compound-matrix formulation.
//
// Both methods solve the same problem, and at low frequency times thickness
// they agree. As f*h grows the direct method's propagated solution comes to be
// dominated by the term growing as exp(+r*d), the decaying part that carries
// the answer falls below float64 precision relative to it, and what is left is
// rounding error. The minor formulation keeps its value of order one.
func TestMinorFormulationSurvivesHighFrequencyThickness(t *testing.T) {
	st := layer.Stack{
		{Thickness: 3, Vp: 400, Vs: 160, Density: 1600},
		{Thickness: 12, Vp: 900, Vs: 400, Density: 2000},
		{Vp: 2200, Vs: 1100, Density: 2400},
	}
	const c = 300.0

	t.Run("they agree while both are viable", func(t *testing.T) {
		for _, f := range []float64{1, 5, 20} {
			m, err := SecularAtVelocity(st, f, c)
			if err != nil {
				t.Fatal(err)
			}
			n, err := SecularNaiveAtVelocity(st, f, c)
			if err != nil {
				t.Fatal(err)
			}
			if math.Signbit(m) != math.Signbit(n) {
				t.Errorf("%g Hz: minors give %g, direct gives %g; they disagree in sign while both are still viable", f, m, n)
			}
		}
	})

	t.Run("the direct method collapses and the minors do not", func(t *testing.T) {
		for _, f := range []float64{200, 400, 800} {
			m, err := SecularAtVelocity(st, f, c)
			if err != nil {
				t.Fatal(err)
			}
			n, err := SecularNaiveAtVelocity(st, f, c)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(n) > 1e-9 {
				t.Errorf("%g Hz (f*h = %g): the direct determinant is %g; it was expected to have collapsed to roundoff",
					f, f*float64(st.TotalThickness()), n)
			}
			if math.Abs(m) < 1e-3 {
				t.Errorf("%g Hz: the minor formulation gave %g; it should still carry a usable value", f, m)
			}
		}
	})
}

func TestRejectsBadArguments(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	if _, err := Secular(st, 0, 1); err == nil {
		t.Error("expected an error for zero frequency")
	}
	if _, err := Secular(st, 1, 0); err == nil {
		t.Error("expected an error for zero wavenumber")
	}
	// Above the half-space shear velocity the mode is not trapped and the
	// decaying subspace does not exist. That is physics, not a failure.
	if _, err := SecularAtVelocity(st, 20, 250); err == nil {
		t.Error("expected an error above the half-space shear velocity")
	}
}
