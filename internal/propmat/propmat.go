// Package propmat computes the Rayleigh secular function of a layered
// half-space: the function whose zeros, in phase velocity at each frequency,
// are the surface-wave modes.
//
// The plan calls a root finder that misses or jumps modes "the worst failure
// mode, because it looks plausible", and the same is true one level down: a
// propagator with a sign error produces a secular function whose zeros are
// smooth, monotone, physically shaped and wrong. So this package is built to
// be checkable at every step rather than transcribed from a reference.
//
// # How it is put together
//
// In each homogeneous layer the motion-stress vector b satisfies db/dz = A b
// with constant A, so propagating across a layer of thickness d is
// multiplication by exp(A d). Rather than transcribing Haskell's closed-form
// layer matrix — four by four of hyperbolic functions, easy to get subtly
// wrong and impossible to check by inspection — A is derived here from the
// equations of motion and the exponential taken numerically. Two properties
// then pin it without any appeal to a reference: A must reproduce the
// constitutive relations under finite differences, and exp(A d) must satisfy
// P(d1)P(d2) = P(d1+d2).
//
// # Why minors
//
// The secular condition is that the free-surface solution, propagated to the
// half-space, lies in the half-space's decaying subspace — a four by four
// determinant. Evaluating it through the propagator product directly is the
// Thomson-Haskell method, and it loses all precision at high frequency times
// thickness: the product accumulates terms growing like exp(+r d) which must
// cancel to leave behaviour going as exp(-r d), and past about fifteen decades
// of growth there is nothing left to cancel with.
//
// Dunkin's fix is to propagate the six two-by-two minors of the solution
// rather than the solution itself. The cancellation then happens once, locally,
// inside each layer's compound matrix, where the terms being subtracted are the
// same size — instead of accumulating across the product. That is V4, and it is
// tested by running both formulations and watching only one survive.
package propmat

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	"geosim.dev/geosim/internal/layer"
)

// SystemMatrix is the dimensionless A in db/dzeta = A b, for the motion-stress
// vector
//
//	b = (u_x, i*u_z, sigma_zx/(muRef*k), i*sigma_zz/(muRef*k))
//
// against dimensionless depth zeta = k*z, at horizontal wavenumber k and
// angular frequency omega, with z increasing downward and everything varying
// as exp(i(k x - omega t)).
//
// Two changes of variable, each earning its place.
//
// The factors of i on the vertical displacement and normal stress make A real.
// They cost nothing — a similarity transform cannot move the secular function's
// zeros — and buy a real four by four instead of a complex one.
//
// The scaling by muRef*k and k is not cosmetic. Written in physical units the
// vector mixes displacements of order one metre with stresses of order 10^8
// pascals, so A spans sixteen orders of magnitude and its exponential is
// hopeless: the first version of this code produced a propagator with
// det P = 1.003 where the trace of A guarantees exactly 1, and P(d1)P(d2)
// differed from P(d1+d2) by a factor of two. Nothing downstream could survive
// that, and at low frequency nothing showed it, because a thin layer's
// propagator is near the identity and hides its own conditioning. Scaled, every
// entry is of order one and both identities hold to roundoff.
//
// Derived, not transcribed. From the constitutive relations
//
//	sigma_zz = i*k*lambda*u_x + (lambda+2mu)*du_z/dz
//	sigma_zx = mu*(du_x/dz + i*k*u_z)
//
// and the equations of motion i*k*sigma_xx + dsigma_zx/dz = -rho*omega^2*u_x,
// i*k*sigma_zx + dsigma_zz/dz = -rho*omega^2*u_z, eliminating sigma_xx with
// sigma_xx = 4*mu*(lambda+mu)/(lambda+2mu)*i*k*u_x + lambda/(lambda+2mu)*sigma_zz.
func SystemMatrix(l layer.Layer, k, omega, muRef float64) *mat.Dense {
	mu := l.ShearModulus()
	lam := l.LameLambda()
	rho := float64(l.Density)
	l2m := lam + 2*mu
	// rho*omega^2/k^2 is rho*c^2: a modulus, so every entry below is a ratio
	// of moduli and dimensionless.
	rc2 := rho * omega * omega / (k * k)

	a := mat.NewDense(4, 4, nil)
	a.Set(0, 1, -1)
	a.Set(0, 2, muRef/mu)

	a.Set(1, 0, lam/l2m)
	a.Set(1, 3, muRef/l2m)

	a.Set(2, 0, (4*mu*(lam+mu)/l2m-rc2)/muRef)
	a.Set(2, 3, -lam/l2m)

	a.Set(3, 1, -rc2/muRef)
	a.Set(3, 2, 1)
	return a
}

// referenceModulus is the stress scale the whole stack is expressed against.
// One value for every layer, because the propagators are multiplied together
// and a per-layer scaling would not compose.
func referenceModulus(s layer.Stack) float64 {
	m := s[0].ShearModulus()
	for _, l := range s {
		m = math.Max(m, l.ShearModulus())
	}
	return m
}

// verticalWavenumbers returns the P and S vertical wavenumbers, as the real
// part that governs growth. A layer in which the wave propagates rather than
// decays vertically contributes no growth at all, so the real part is zero
// there and the scaling below is a no-op.
func verticalWavenumbers(l layer.Layer, k, omega float64) (ra, rb float64) {
	sq := func(v float64) float64 {
		return math.Sqrt(math.Max(0, k*k-omega*omega/(v*v)))
	}
	return sq(float64(l.Vp)), sq(float64(l.Vs))
}

// Propagator is exp(A d) for one layer, factored as exp(scale) * Bounded.
//
// The factoring is what keeps the arithmetic in range. exp(A d) has entries
// growing like exp(r_alpha d), which at a hundred hertz through a ten metre
// layer is already e^40; squaring that in the compound matrix would reach e^80
// and the useful information would be sixteen decades below the leading term.
// Subtracting the largest eigenvalue from A before exponentiating leaves every
// eigenvalue with non-positive real part, so Bounded has entries of order one.
//
// The scale is tracked separately and, in the end, discarded: it is a positive
// factor common to the whole secular function, and multiplying a function by a
// positive number does not move its zeros.
type Propagator struct {
	Bounded *mat.Dense
	Scale   float64 // the log of the factored-out exp(lambda_max * d)
}

// LayerPropagator builds the propagator for one layer.
func LayerPropagator(l layer.Layer, k, omega, muRef float64) Propagator {
	// Dimensionless thickness, to match the dimensionless system matrix.
	zeta := k * float64(l.Thickness)
	ra, _ := verticalWavenumbers(l, k, omega)
	// r_alpha is always the larger, since Vp exceeds Vs, so it alone sets the
	// growth. In dimensionless depth the eigenvalue is r_alpha/k.
	lambdaMax := ra / k

	a := SystemMatrix(l, k, omega, muRef)
	shifted := mat.NewDense(4, 4, nil)
	shifted.Scale(zeta, a)
	for i := range 4 {
		shifted.Set(i, i, shifted.At(i, i)-lambdaMax*zeta)
	}

	p := mat.NewDense(4, 4, nil)
	p.Exp(shifted)
	return Propagator{Bounded: p, Scale: lambdaMax * zeta}
}

// pairs indexes the six two-by-two minors of a four-row matrix, and comp gives
// each pair's complement within {0,1,2,3}.
var (
	pairs = [6][2]int{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}
	comp  = [6]int{5, 4, 3, 2, 1, 0}
	// laplaceSign is (-1)^(i+j+1) for the generalised Laplace expansion of a
	// four by four determinant along its first two columns.
	laplaceSign = [6]float64{1, -1, 1, 1, -1, 1}
)

// Compound is the second compound matrix of p: the six by six matrix that
// propagates two-by-two minors the way p propagates the vectors themselves.
//
// This is the whole point of the method. Each entry is a difference of two
// products of p's entries, and because p is bounded those two products are the
// same size — so the cancellation is benign and happens once. Propagating the
// minors through these is stable where propagating the vectors is not.
func Compound(p *mat.Dense) *mat.Dense {
	c := mat.NewDense(6, 6, nil)
	for r, ij := range pairs {
		i, j := ij[0], ij[1]
		for col, kl := range pairs {
			kk, ll := kl[0], kl[1]
			c.Set(r, col, p.At(i, kk)*p.At(j, ll)-p.At(i, ll)*p.At(j, kk))
		}
	}
	return c
}

// HalfSpaceModes returns the two motion-stress vectors of the half-space's
// decaying solutions: the evanescent P wave and the evanescent SV wave.
//
// In closed form, deliberately. The first attempt took them from a numerical
// eigendecomposition, which produced a secular function that flipped sign
// discontinuously at velocities where nothing physical happens — the solver
// was free to return the two eigenvectors in either order and with either
// sign, and the two-by-two minor built from them inherited that. It looked
// exactly like an extra root. Closed forms vary continuously with velocity by
// construction, so the question cannot arise.
//
// Derived from the evanescent potentials. For P, phi = exp(-r_alpha z) gives
// u_x = ik*phi and u_z = -r_alpha*phi; for SV, psi = exp(-r_beta z) gives
// u_x = r_beta*psi and u_z = ik*psi. Feeding those through the constitutive
// relations and the (u_x, i*u_z, sigma_zx, i*sigma_zz) convention leaves both
// vectors real.
//
// The check that they are right is not that the algebra above reads well: for
// a half-space with no layers the secular function reduces to the two-by-two
// minor of these vectors' traction rows, and that reduces in turn to
// (2 - c^2/beta^2)^2 = 4*sqrt(1-c^2/alpha^2)*sqrt(1-c^2/beta^2) — the
// classical Rayleigh equation slice 0 already solves independently.
func HalfSpaceModes(l layer.Layer, k, omega, muRef float64) (p, sv [4]float64, err error) {
	if omega/k >= float64(l.Vs) {
		return p, sv, fmt.Errorf("propmat: phase velocity %g m/s is not below the half-space shear velocity %g; the mode is not trapped",
			omega/k, l.Vs)
	}
	ra, rb := verticalWavenumbers(l, k, omega)
	mu := l.ShearModulus()
	rho := float64(l.Density)
	rc2 := rho * omega * omega / (k * k)
	a, b := ra/k, rb/k

	// Both expressed in the same dimensionless variables as SystemMatrix, and
	// each normalised so its displacement components are of order one.
	p = [4]float64{1, -a, -2 * (mu / muRef) * a, (2*mu - rc2) / muRef}
	sv = [4]float64{b, -1, (rc2 - 2*mu) / muRef, 2 * (mu / muRef) * b}
	return p, sv, nil
}

// halfSpaceMinors is the six two-by-two minors of the four by two matrix whose
// columns are the half-space's decaying solutions.
//
// A trapped mode has to die away with depth, so only these two are admissible;
// the growing pair would be a wave arriving from infinity. This is only well
// posed while the phase velocity stays below the half-space's shear velocity —
// above it the wave radiates into the half-space rather than being trapped.
// That is a real physical boundary, not a limitation of the method, and it is
// where the search for modes has to stop.
func halfSpaceMinors(l layer.Layer, k, omega, muRef float64) ([6]float64, error) {
	var m [6]float64
	p, sv, err := HalfSpaceModes(l, k, omega, muRef)
	if err != nil {
		return m, err
	}
	for n, ij := range pairs {
		i, j := ij[0], ij[1]
		m[n] = p[i]*sv[j] - p[j]*sv[i]
	}
	return m, nil
}

// Secular evaluates the Rayleigh dispersion function at angular frequency
// omega and horizontal wavenumber k. Its zeros in phase velocity are the modes.
//
// The value returned is normalised — its magnitude carries no meaning, only its
// sign and where it crosses. Every layer's growth factor is divided out, and
// the minor vector is rescaled as it propagates, because the useful quantity
// spans far more decades than a float64 can hold and none of that dynamic range
// affects where the function is zero.
func Secular(s layer.Stack, omega, k float64) (float64, error) {
	if k <= 0 || omega <= 0 {
		return 0, fmt.Errorf("propmat: omega and k must be positive, got %g and %g", omega, k)
	}
	// Free surface: zero traction, so the solution starts in the span of the
	// first two basis vectors. The minor vector of that two-plane is (1,0,...).
	m := [6]float64{1, 0, 0, 0, 0, 0}
	muRef := referenceModulus(s)

	for _, l := range s[:len(s)-1] {
		p := LayerPropagator(l, k, omega, muRef)
		c := Compound(p.Bounded)

		var next [6]float64
		for r := range 6 {
			var sum float64
			for col := range 6 {
				sum += c.At(r, col) * m[col]
			}
			next[r] = sum
		}
		// Renormalise. The scale is a positive common factor and cannot move a
		// zero, but without removing it the vector overflows within a few
		// layers at seismic frequencies.
		var norm float64
		for _, v := range next {
			norm = math.Max(norm, math.Abs(v))
		}
		if norm == 0 {
			return 0, fmt.Errorf("propmat: minor vector collapsed to zero at omega=%g k=%g", omega, k)
		}
		for i := range next {
			next[i] /= norm
		}
		m = next
	}

	e, err := halfSpaceMinors(s.HalfSpace(), k, omega, muRef)
	if err != nil {
		return 0, err
	}

	// Generalised Laplace expansion of det[B | E] along the first two columns:
	// each minor of the propagated solution pairs with the complementary minor
	// of the half-space subspace.
	var f float64
	for n := range 6 {
		f += laplaceSign[n] * m[n] * e[comp[n]]
	}
	return f, nil
}

// SecularAtVelocity is Secular expressed in the variable the modes are usually
// quoted in.
func SecularAtVelocity(s layer.Stack, freq, c float64) (float64, error) {
	omega := 2 * math.Pi * freq
	return Secular(s, omega, omega/c)
}

// SecularNaive evaluates the same dispersion function by propagating the
// solution itself rather than its minors — the plain Thomson-Haskell method.
//
// It exists to be compared against, not to be used. At low frequency times
// thickness it agrees with Secular to roundoff, which is worth having as a
// check that the minor recursion is solving the same problem. As f*h grows it
// falls apart, and the way it falls apart is the argument for the whole
// compound-matrix apparatus: the propagated solution is dominated by the term
// growing as exp(+r*d), the physically relevant decaying part sits sixteen
// decades below it, and once the ratio exceeds what a float64 can represent
// the decaying part is simply gone. The determinant then measures rounding
// error, and its zeros are wherever that noise happens to change sign.
//
// This is the difference between a dispersion curve and a plausible-looking
// fiction, which is why the plan calls the Dunkin formulation non-optional.
func SecularNaive(s layer.Stack, omega, k float64) (float64, error) {
	if k <= 0 || omega <= 0 {
		return 0, fmt.Errorf("propmat: omega and k must be positive, got %g and %g", omega, k)
	}
	muRef := referenceModulus(s)

	// The free-surface solution space: zero traction, so the first two basis
	// vectors span it.
	b := mat.NewDense(4, 2, []float64{1, 0, 0, 1, 0, 0, 0, 0})

	for _, l := range s[:len(s)-1] {
		p := LayerPropagator(l, k, omega, muRef)
		var next mat.Dense
		next.Mul(p.Bounded, b)
		// Same renormalisation the minor recursion uses, so the comparison
		// isolates the formulation rather than the bookkeeping.
		var norm float64
		for i := range 4 {
			for j := range 2 {
				norm = math.Max(norm, math.Abs(next.At(i, j)))
			}
		}
		if norm == 0 {
			return 0, fmt.Errorf("propmat: solution collapsed at omega=%g k=%g", omega, k)
		}
		next.Scale(1/norm, &next)
		b = &next
	}

	p, sv, err := HalfSpaceModes(s.HalfSpace(), k, omega, muRef)
	if err != nil {
		return 0, err
	}
	full := mat.NewDense(4, 4, nil)
	for i := range 4 {
		full.Set(i, 0, b.At(i, 0))
		full.Set(i, 1, b.At(i, 1))
		full.Set(i, 2, p[i])
		full.Set(i, 3, sv[i])
	}
	return mat.Det(full), nil
}

// SecularNaiveAtVelocity is SecularNaive in the variable modes are quoted in.
func SecularNaiveAtVelocity(s layer.Stack, freq, c float64) (float64, error) {
	omega := 2 * math.Pi * freq
	return SecularNaive(s, omega, omega/c)
}
