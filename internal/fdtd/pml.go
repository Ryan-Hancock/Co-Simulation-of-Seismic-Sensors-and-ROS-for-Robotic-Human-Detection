package fdtd

import "math"

// profile is one direction's convolutional PML coefficients, indexed by the
// grid index of the field the derivative belongs to.
//
// Outside the absorbing layer a is zero and b is one, so the recursion holds
// the memory variable at zero and contributes nothing. That is why the interior
// update below can be written once, with no branch on whether a cell is inside
// the layer: correctness does not depend on the loop bounds, only cost does.
type profile struct {
	a, b  []float64
	first int // lowest index with any damping, for restricting the loops
}

// newProfile builds the damping profile for one direction.
//
// The C-PML of Roden and Gedney, with the grid stretch kappa left at one. The
// stretch helps a classical PML with evanescent fields; the frequency shift
// alpha helps far more, and it is the frequency shift that makes this layer
// able to absorb a Rayleigh wave — which arrives at the outer boundary
// travelling along it, at grazing incidence, where an unshifted PML reflects
// badly.
//
// n is the number of nodes, offset is the node's position within a cell (0 for
// integer positions, 0.5 for half), inner is the coordinate where the layer
// begins, and thickness its extent.
func newProfile(n int, h, offset, inner, thickness float64, vMax, reflection, dominant, dt float64) profile {
	p := profile{a: make([]float64, n), b: make([]float64, n), first: n}
	for i := range p.b {
		p.b[i] = 1
	}
	if thickness <= 0 {
		return p
	}
	// Quadratic grading. d0 comes from requiring that a wave crossing the
	// layer and returning is attenuated to the stated reflection coefficient:
	// integral of d over the layer is -(N+1)*v*ln(R)/2 for a profile of
	// order N.
	const order = 2
	d0 := -(order + 1) * vMax * math.Log(reflection) / (2 * thickness)
	aMax := math.Pi * dominant
	for i := range n {
		s := (float64(i)+offset)*h - inner
		if s <= 0 {
			continue
		}
		x := math.Min(s/thickness, 1)
		d := d0 * math.Pow(x, order)
		alpha := aMax * (1 - x)
		if d+alpha <= 0 {
			continue
		}
		b := math.Exp(-(d + alpha) * dt)
		p.a[i] = d * (b - 1) / (d + alpha)
		p.b[i] = b
		if i < p.first {
			p.first = i
		}
	}
	return p
}

// update advances one memory variable and returns the correction to add to the
// plain derivative.
func (p *profile) update(psi *float64, i int, deriv float64) float64 {
	*psi = p.b[i]**psi + p.a[i]*deriv
	return *psi
}
