// Package fk computes the response of a layered half-space to a point force on
// its surface, by integrating over horizontal wavenumber.
//
// This is what slice 3 exists for. Slice 0's model kept only the far-field
// Rayleigh pole, and the plan is explicit about why that is not enough here:
// the asymptotic needs r >> lambda, and at 20 Hz over soil with cR = 150 m/s
// the wavelength is 7.5 m, so a robot detecting a person at 5 m is inside one
// wavelength. Near-field terms and body-wave arrivals are not negligible there,
// and a surface-wave-only model is quietly wrong in exactly the regime O3
// operates in.
//
// Integrating over all wavenumbers rather than summing residues at the poles
// gives the whole field at once — near field, body waves, every Rayleigh mode,
// and the static term — with no decision about which contributions to keep.
//
// # Why this is complex arithmetic throughout
//
// A lossless medium puts the Rayleigh pole on the real wavenumber axis, and the
// integral through it does not converge. Attenuation moves it off, which is
// both the numerical fix and the physics: real ground is lossy. So layer
// velocities here are complex, following the same Kjartansson constant-Q model
// the homogeneous solver uses, and Q is not an optional refinement but what
// makes the calculation well posed.
package fk

import (
	"fmt"
	"math"
	"math/cmplx"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/units"
)

// Medium is a layered half-space with attenuation.
type Medium struct {
	Stack layer.Stack
	// RefFreq anchors the constant-Q dispersion: the frequency at which each
	// layer's velocity equals its nominal value. Zero uses 30 Hz.
	RefFreq units.Hertz
	// DefaultQ is used for layers that do not set one. Zero uses 30.
	DefaultQ float64
}

const (
	defaultRefFreq = 30.0
	defaultQ       = 30.0
)

func (m Medium) refFreq() float64 {
	if m.RefFreq > 0 {
		return float64(m.RefFreq)
	}
	return defaultRefFreq
}

func (m Medium) qOf(l layer.Layer) float64 {
	if l.Qs > 0 {
		return l.Qs
	}
	if m.DefaultQ > 0 {
		return m.DefaultQ
	}
	return defaultQ
}

// complexVelocity applies Kjartansson's constant-Q model: c(omega) =
// c0*(i*omega/omega0)^gamma with gamma = arctan(1/Q)/pi.
//
// The same model the homogeneous solver uses, and for the same reason — it is
// exactly causal, where the more commonly seen logarithmic pairing is causal
// only to first order in 1/Q and needs an arbitrary low-frequency clamp.
func (m Medium) complexVelocity(v float64, q, freq float64) complex128 {
	if q <= 0 || freq <= 0 {
		return complex(v, 0)
	}
	gam := math.Atan(1/q) / math.Pi
	c0 := v * math.Cos(math.Pi*gam/2)
	return complex(c0, 0) * cmplx.Pow(complex(freq/m.refFreq(), 0), complex(gam, 0)) *
		cmplx.Exp(complex(0, math.Pi*gam/2))
}

// props are one layer's complex elastic constants at a frequency.
type props struct {
	mu, lambda  complex128
	rho         float64
	alpha, beta complex128
}

func (m Medium) propsOf(l layer.Layer, freq float64) props {
	q := m.qOf(l)
	// P attenuation follows from putting the loss in shear: Qp = 9/4 * Qs.
	a := m.complexVelocity(float64(l.Vp), 2.25*q, freq)
	b := m.complexVelocity(float64(l.Vs), q, freq)
	rho := float64(l.Density)
	mu := complex(rho, 0) * b * b
	lam := complex(rho, 0)*a*a - 2*mu
	return props{mu: mu, lambda: lam, rho: rho, alpha: a, beta: b}
}

// eigen returns the four eigenvalues and eigenvectors of the dimensionless
// system matrix, in the order (P down, SV down, P up, SV up).
//
// Written out rather than found numerically. The vectors are the evanescent and
// propagating P and SV solutions, derived from their potentials; a numerical
// eigensolver would return them in an arbitrary order with arbitrary signs,
// which in the dispersion solver produced a secular function that flipped sign
// where nothing physical happened.
func (p props) eigen(k, omega, muRef float64) (lam [4]complex128, vec cmat) {
	kc := complex(k, 0)
	w2 := complex(omega*omega, 0)
	// Vertical wavenumbers, made dimensionless by k. The branch with positive
	// real part is the one that decays downward.
	ra := cmplx.Sqrt(1 - w2/(kc*kc*p.alpha*p.alpha))
	rb := cmplx.Sqrt(1 - w2/(kc*kc*p.beta*p.beta))
	if real(ra) < 0 {
		ra = -ra
	}
	if real(rb) < 0 {
		rb = -rb
	}
	rc2 := complex(p.rho, 0) * w2 / (kc * kc) // rho*c^2, a modulus
	mr := complex(muRef, 0)
	mu := p.mu

	lam = [4]complex128{-ra, -rb, ra, rb}

	// P down / up: components 1 and 2 change sign with the direction.
	pd := cvec{1, -ra, -2 * mu / mr * ra, (2*mu - rc2) / mr}
	pu := cvec{1, ra, 2 * mu / mr * ra, (2*mu - rc2) / mr}
	// SV down / up: components 0 and 3 change sign.
	sd := cvec{rb, -1, (rc2 - 2*mu) / mr, 2 * mu / mr * rb}
	su := cvec{-rb, -1, (rc2 - 2*mu) / mr, -2 * mu / mr * rb}

	for i := range 4 {
		vec[i][0], vec[i][1], vec[i][2], vec[i][3] = pd[i], sd[i], pu[i], su[i]
	}
	return lam, vec
}

// propagator is exp(A * k * d) for one layer, with the largest growing
// exponential factored out so every entry stays of order one.
//
// The factored scale is discarded rather than tracked: it multiplies the whole
// propagated solution, and the boundary-value problem below is homogeneous in
// that solution except for the source term, which is applied at the surface
// where the scale is one. Keeping it would only risk overflow.
func (p props) propagator(k, omega, muRef, thickness float64) (cmat, bool) {
	lam, e := p.eigen(k, omega, muRef)
	zeta := complex(k*thickness, 0)

	// The dominant growth, subtracted before exponentiating.
	var maxRe float64
	for _, l := range lam {
		maxRe = math.Max(maxRe, real(l))
	}
	var d cmat
	for i := range 4 {
		d[i][i] = cmplx.Exp((lam[i] - complex(maxRe, 0)) * zeta)
	}

	inv, ok := invert(e)
	if !ok {
		return cmat{}, false
	}
	return e.mul(d).mul(inv), true
}

// invert is a 4x4 complex inverse by solving against the identity.
func invert(a cmat) (cmat, bool) {
	var out cmat
	for j := range 4 {
		var e cvec
		e[j] = 1
		col, ok := a.solve(e)
		if !ok {
			return cmat{}, false
		}
		for i := range 4 {
			out[i][j] = col[i]
		}
	}
	return out, true
}

// referenceModulus is the stress scale the whole stack is expressed against.
func (m Medium) referenceModulus() float64 {
	r := m.Stack[0].ShearModulus()
	for _, l := range m.Stack {
		r = math.Max(r, l.ShearModulus())
	}
	return r
}

// SurfaceResponse is the wavenumber-domain vertical displacement at the free
// surface, per newton of vertical point force there, at wavenumber k and
// frequency f. Units are metres per newton per unit of the Hankel measure.
//
// The boundary-value problem is: prescribed traction at the surface, only
// decaying solutions in the half-space. Four unknowns — the two surface
// displacement components and the two half-space amplitudes — and four
// equations.
func (m Medium) SurfaceResponse(k, freq float64) (complex128, error) {
	if k <= 0 || freq <= 0 {
		return 0, fmt.Errorf("fk: wavenumber and frequency must be positive, got %g and %g", k, freq)
	}
	if err := m.Stack.Validate(); err != nil {
		return 0, err
	}
	omega := 2 * math.Pi * freq
	muRef := m.referenceModulus()

	// Propagate from the surface down to the top of the half-space.
	var total cmat
	for i := range 4 {
		total[i][i] = 1
	}
	for _, l := range m.Stack[:len(m.Stack)-1] {
		p, ok := m.propsOf(l, freq).propagator(k, omega, muRef, float64(l.Thickness))
		if !ok {
			return 0, fmt.Errorf("fk: layer propagator is singular at k=%g f=%g", k, freq)
		}
		total = p.mul(total)
	}

	// The half-space's two decaying solutions.
	_, e := m.propsOf(m.Stack.HalfSpace(), freq).eigen(k, omega, muRef)
	down0, down1 := e.column(0), e.column(1)

	// A vertical point force F on the surface has Hankel-transformed normal
	// traction -F/(2*pi); the shear traction is zero. In the scaled vector
	// b4 = i*sigma_zz/(muRef*k), so a unit force gives b4 = -i/(2*pi*muRef*k).
	b4 := complex(0, -1) / complex(2*math.Pi*muRef*k, 0)

	// Unknowns: (u_x0, i*u_z0, halfspace P amplitude, halfspace SV amplitude).
	var a cmat
	c0, c1, c3 := total.column(0), total.column(1), total.column(3)
	for i := range 4 {
		a[i][0] = c0[i]
		a[i][1] = c1[i]
		a[i][2] = -down0[i]
		a[i][3] = -down1[i]
	}
	var rhs cvec
	for i := range 4 {
		rhs[i] = -b4 * c3[i]
	}
	x, ok := a.solve(rhs)
	if !ok {
		return 0, fmt.Errorf("fk: surface system is singular at k=%g f=%g", k, freq)
	}
	// b2 = i*u_z, so u_z = -i*b2.
	return complex(0, -1) * x[1], nil
}

// Integration configures the wavenumber quadrature.
type Integration struct {
	// Samples is how many wavenumbers the integral uses. The integrand
	// oscillates as J0(k*r) and has a pole of width about k_R/(2Q) near the
	// Rayleigh wavenumber, so this has to resolve the narrower of the two.
	// Zero uses 20000.
	Samples int
	// KMaxFactor sets how far the integral runs, as a multiple of the largest
	// body-wave wavenumber. The integrand is evanescent beyond that and decays,
	// but the near-field terms live there and truncating too early removes
	// exactly the contribution this package exists to capture. Zero uses 30.
	KMaxFactor float64
}

func (o Integration) samples() int {
	if o.Samples > 0 {
		return o.Samples
	}
	return 20000
}

func (o Integration) kMaxFactor() float64 {
	if o.KMaxFactor > 0 {
		return o.KMaxFactor
	}
	return 30
}

// StaticCoefficient is the limit of k times the wavenumber-domain response as
// k grows: the coefficient of the medium's static, near-field behaviour.
//
// For a homogeneous half-space it is exactly Boussinesq's (1-nu)/(2*pi*mu), and
// for a layered one it is the same expression in the *surface* layer's
// constants, because short wavelengths only ever sample the top. Evaluated
// numerically rather than assumed, so it stays right when the moduli are
// complex.
func (m Medium) StaticCoefficient(freq float64, kFar float64) (complex128, error) {
	u, err := m.SurfaceResponse(kFar, freq)
	if err != nil {
		return 0, err
	}
	return complex(kFar, 0) * u, nil
}

// VerticalDisplacement is the vertical surface displacement at range r per
// newton of vertical point force at the origin, at frequency f, in m/N.
//
//	u_z(r) = integral of uTilde(k) * J0(k*r) * k dk
//
// Everything is in this integral: the Rayleigh poles as sharp peaks near
// k = omega/c, the body-wave branch contributions, and the near-field terms at
// large k that no residue sum contains.
//
// # Why the static part is subtracted rather than integrated
//
// The response tends to C/k as k grows, so the integrand tends to C*J0(k*r),
// whose integral converges only as the square root of the truncation point:
// cutting at k*r = 40 still leaves a 13% error, and pushing that to a
// tolerable level by brute force would need wavenumbers orders of magnitude
// further out. Subtracting the asymptote and adding back its exact integral,
//
//	integral of C*J0(k*r) dk from 0 to infinity = C/r,
//
// leaves a remainder that decays properly and converges in a few thousand
// samples. It also makes the static limit exact by construction, which is the
// right behaviour: at zero frequency the answer must be Boussinesq's, and now
// it is, rather than being approached from a truncated oscillation.
func (m Medium) VerticalDisplacement(r units.Metres, freq float64, opt Integration) (complex128, error) {
	if r <= 0 {
		return 0, fmt.Errorf("fk: range must be positive, got %g m", r)
	}
	if freq <= 0 {
		return 0, fmt.Errorf("fk: frequency must be positive, got %g Hz", freq)
	}
	slowest, _ := m.Stack.VelocityBounds()
	kBody := 2 * math.Pi * freq / float64(slowest)
	kMax := opt.kMaxFactor() * kBody
	// At short range the near field reaches wavenumbers well past the
	// body-wave branch, so the cut has to follow 1/r too.
	kMax = math.Max(kMax, 40/float64(r))

	// Far enough out to be on the asymptote, which the response reaches once k
	// is well past the body-wave wavenumbers.
	c, err := m.StaticCoefficient(freq, math.Max(200*kBody, 400/float64(r)))
	if err != nil {
		return 0, err
	}

	n := opt.samples()
	dk := kMax / float64(n)

	// Trapezoidal on the remainder. At k -> 0 the integrand k*uTilde - C tends
	// to -C, finite, so the lower endpoint is well behaved.
	var sum complex128
	for i := 1; i <= n; i++ {
		k := float64(i) * dk
		u, err := m.SurfaceResponse(k, freq)
		if err != nil {
			return 0, err
		}
		w := 1.0
		if i == n {
			w = 0.5
		}
		rem := complex(k, 0)*u - c
		sum += complex(w*math.J0(k*float64(r))*dk, 0) * rem
	}
	return c/complex(float64(r), 0) + sum, nil
}
