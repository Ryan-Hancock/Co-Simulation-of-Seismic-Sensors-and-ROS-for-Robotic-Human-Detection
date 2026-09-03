// Package fdtd is the independent numerical route to the same answer.
//
// internal/fk solves the layered problem in the frequency-wavenumber domain:
// propagator matrices, a boundary-value solve at each wavenumber, a Hankel
// quadrature. It agrees with the analytic half-space and with disba, but both
// of those checks share assumptions with it — disba solves the same secular
// equation, and the analytic model is the same physics in closed form. None of
// them would catch a sign error in the layer propagator that happened to be
// consistent across the whole family.
//
// This package shares no code with any of them. It integrates the elastodynamic
// equations forward in time on a grid: no wavenumbers, no propagators, no
// Hankel transform, no secular function. Where the two agree, the agreement is
// evidence; where they disagree, one of them is wrong and the difference says
// where to look. That is validation V5, and with V2 it is what the layered
// claim rests on.
//
// Axisymmetric, because a vertical point force on a horizontally layered
// half-space has no azimuthal dependence. That is not a simplification of the
// physics for this source — it is exact, and it turns a 3-D grid into a 2-D
// one, which is the difference between a check that runs in seconds and one
// that runs overnight. The cost is that it can never model the lateral
// heterogeneity or topography that WP2 will want; this is L2 in the hierarchy,
// not L3.
package fdtd

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/units"
	"geosim.dev/geosim/internal/visco"
)

// Model is the medium and the grid it is discretised on.
type Model struct {
	// Stack is the layered medium. Layers are horizontal, so every material
	// property varies with depth alone — which is why the absorbing boundary
	// at the outer radius sees exactly the medium it is continuing.
	Stack layer.Stack
	// Relax is the attenuation. The zero value, with both relaxation times
	// zero, is elastic.
	Relax visco.SLS
	// RefFreq is where each layer's nominal velocity is its phase velocity.
	// Zero uses 30 Hz, matching internal/fk.
	RefFreq units.Hertz

	// MaxRange is the radial extent of the interior, beyond which the
	// absorbing layer begins. Receivers must lie inside it with room to
	// spare: a receiver near the boundary sees whatever the boundary fails
	// to absorb.
	MaxRange units.Metres
	// Depth is the vertical extent of the interior.
	Depth units.Metres
	// Spacing is the cell size, equal in both directions. SpacingFor picks
	// one from a bandwidth.
	Spacing units.Metres

	// PMLCells is the thickness of the absorbing layer in cells. Zero uses
	// 20, which measurement puts at around -60 dB.
	PMLCells int
	// PMLReflection is the theoretical reflection coefficient the damping
	// profile is designed for. Zero uses 1e-4. Making it smaller does not
	// monotonically help: a steeper profile reflects more off its own
	// gradient than it absorbs.
	PMLReflection float64
	// DominantFreq tunes the C-PML frequency shift, which is what stops the
	// layer reflecting grazing and evanescent energy — the Rayleigh wave, in
	// other words, which arrives at the outer boundary travelling exactly
	// along it. Zero picks the middle of the well-resolved band.
	DominantFreq units.Hertz

	// Courant scales the time step below the stability limit. Zero uses 0.5.
	Courant float64
}

const (
	defaultRefFreq  = 30.0
	defaultPMLCells = 20
	defaultPMLRefl  = 1e-4
	defaultCourant  = 0.5
)

func (m Model) refFreq() float64 {
	if m.RefFreq > 0 {
		return float64(m.RefFreq)
	}
	return defaultRefFreq
}

func (m Model) pmlCells() int {
	if m.PMLCells > 0 {
		return m.PMLCells
	}
	return defaultPMLCells
}

func (m Model) pmlReflection() float64 {
	if m.PMLReflection > 0 {
		return m.PMLReflection
	}
	return defaultPMLRefl
}

func (m Model) courant() float64 {
	if m.Courant > 0 {
		return m.Courant
	}
	return defaultCourant
}

// SpacingFor is the cell size that puts the given number of cells across the
// shortest wavelength in the band.
//
// The shortest wavelength is the slowest shear velocity over the highest
// frequency, and it is always the shear wave that binds — the P wave is faster
// and therefore longer. A second-order scheme wants a lot of cells: ten is the
// usual rule of thumb and gives percent-level phase error, twenty is where the
// error stops dominating a comparison.
func (m Model) SpacingFor(maxFreq units.Hertz, cellsPerWavelength float64) (units.Metres, error) {
	if maxFreq <= 0 || cellsPerWavelength <= 0 {
		return 0, fmt.Errorf("fdtd: bandwidth and cells per wavelength must be positive")
	}
	if err := m.Stack.Validate(); err != nil {
		return 0, err
	}
	slowest, _ := m.Stack.VelocityBounds()
	// Attenuation slows the relaxed medium below its nominal velocity, so
	// take the slowest velocity actually present rather than the label.
	v := float64(slowest)
	if m.Relax.TauEps > m.Relax.TauSigma {
		v = m.Relax.RelaxedVelocity(v, m.refFreq())
	}
	return units.Metres(v / (float64(maxFreq) * cellsPerWavelength)), nil
}

// material holds the depth-varying constants, one entry per row.
//
// Only depth varies, so these are one-dimensional even though the grid is two
// dimensional — which also means the outer absorbing layer continues the same
// layering it abuts, rather than approximating it.
type material struct {
	// rho and lamU, muU are at integer depths, where the normal stresses sit.
	rho, lamU, muU []float64
	// rhoZ is the density at half-depths, where the vertical velocity sits:
	// an arithmetic average across an interface, which is what makes the
	// interface a second-order feature of the scheme rather than a
	// first-order one.
	rhoZ []float64
	// muZ is the shear modulus at half-depths, where the shear stress sits.
	// Harmonic, not arithmetic: the shear stress is continuous across an
	// interface and the strain is not, so it is the compliance that averages.
	muZ []float64
}

func (m Model) materials(nz int, dz float64) material {
	mat := material{
		rho:  make([]float64, nz),
		lamU: make([]float64, nz),
		muU:  make([]float64, nz),
		rhoZ: make([]float64, nz),
		muZ:  make([]float64, nz),
	}
	fRef := m.refFreq()
	lossy := m.Relax.TauEps > m.Relax.TauSigma
	for j := range nz {
		l := m.layerAt(float64(j) * dz)
		rho := float64(l.Density)
		vp, vs := float64(l.Vp), float64(l.Vs)
		var mp, ms float64
		if lossy {
			_, mp = m.Relax.Moduli(vp, fRef, rho)
			_, ms = m.Relax.Moduli(vs, fRef, rho)
		} else {
			mp, ms = rho*vp*vp, rho*vs*vs
		}
		mat.rho[j] = rho
		mat.muU[j] = ms
		mat.lamU[j] = mp - 2*ms
	}
	for j := range nz {
		k := min(j+1, nz-1)
		mat.rhoZ[j] = 0.5 * (mat.rho[j] + mat.rho[k])
		mat.muZ[j] = harmonic(mat.muU[j], mat.muU[k])
	}
	return mat
}

func harmonic(a, b float64) float64 {
	if a <= 0 || b <= 0 {
		return 0
	}
	return 2 * a * b / (a + b)
}

// layerAt is the layer containing a depth. A node exactly on an interface is
// assigned to the layer below, so a stated thickness is the depth of the last
// row belonging to the layer above.
func (m Model) layerAt(z float64) layer.Layer {
	var top float64
	for _, l := range m.Stack[:len(m.Stack)-1] {
		top += float64(l.Thickness)
		if z < top {
			return l
		}
	}
	return m.Stack.HalfSpace()
}

// timeStep is the plain two-dimensional stability limit, h/(vp*sqrt(2)),
// before the Courant factor is applied.
//
// The axis is stiffer than the interior: the geometric terms there carry a
// factor of two relative to an ordinary difference, so the true limit is
// somewhat below this. The default Courant factor covers the gap, and
// TestStabilityLimit pins where the scheme actually diverges rather than
// trusting the argument.
// maxVelocity is the fastest wave the scheme can carry: the unrelaxed
// compressional velocity of the stiffest layer.
//
// The unrelaxed modulus, not the nominal one. A viscoelastic medium is
// stiffest on the instant a stress is applied and relaxes from there, so a
// step chosen from the reference-frequency velocity would be marginally
// unstable at high Q and comfortably unstable at low Q.
func (m Model) maxVelocity() float64 {
	vp := 0.0
	fRef := m.refFreq()
	lossy := m.Relax.TauEps > m.Relax.TauSigma
	for _, l := range m.Stack {
		v := float64(l.Vp)
		if lossy {
			// The unrelaxed modulus is the stiffest the medium ever is, and
			// stability is set by the fastest wave the scheme can carry, not
			// by the one it carries at the reference frequency.
			c0 := m.Relax.RelaxedVelocity(v, fRef)
			v = c0 * math.Sqrt(1+m.Relax.Delta())
		}
		vp = math.Max(vp, v)
	}
	return vp
}

func (m Model) timeStep(h float64) float64 {
	return h / (m.maxVelocity() * math.Sqrt2)
}
