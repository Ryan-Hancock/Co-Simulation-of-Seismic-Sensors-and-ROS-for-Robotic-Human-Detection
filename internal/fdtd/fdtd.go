package fdtd

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/units"
)

// Sim is a running simulation: the grid, the fields on it, and the position in
// time.
//
// The staggering is Virieux's, carried over to cylindrical coordinates. Normal
// stresses sit at integer radius and integer depth; the radial velocity a half
// cell out in radius; the vertical velocity a half cell down; the shear stress
// half a cell in both. Velocities live on half time steps and stresses on whole
// ones. Nothing is ever interpolated to make an update work, which is what
// makes the scheme second order and, more usefully, what makes each equation
// checkable on its own.
type Sim struct {
	m          Model
	nr, nz     int
	dr, dz, dt float64
	mat        material

	vr, vz             []float64
	trr, tqq, tzz, trz []float64

	lossy              bool
	memDecay, memDrive float64
	rrr, rqq, rzz, rrz []float64

	pRint, pRhalf, pZint, pZhalf profile
	psiTrrR, psiTrzZ             []float64 // radial velocity
	psiTrzR, psiTzzZ             []float64 // vertical velocity
	psiVrR, psiVzZ               []float64 // normal stresses
	psiVzR, psiVrZ               []float64 // shear stress

	force   []float64
	srcArea float64
	step    int
}

// New allocates a simulation.
func New(m Model) (*Sim, error) {
	if err := m.Stack.Validate(); err != nil {
		return nil, err
	}
	if m.Spacing <= 0 {
		return nil, fmt.Errorf("fdtd: spacing must be positive, got %g m", m.Spacing)
	}
	if m.MaxRange <= 0 || m.Depth <= 0 {
		return nil, fmt.Errorf("fdtd: domain must be positive, got %g m by %g m", m.MaxRange, m.Depth)
	}
	if m.Relax.TauEps < m.Relax.TauSigma {
		return nil, fmt.Errorf("fdtd: TauEps must be at least TauSigma for a dissipative solid")
	}
	h := float64(m.Spacing)
	pml := m.pmlCells()
	nr := int(math.Round(float64(m.MaxRange)/h)) + 1 + pml
	nz := int(math.Round(float64(m.Depth)/h)) + 1 + pml

	s := &Sim{m: m, nr: nr, nz: nz, dr: h, dz: h}
	s.dt = m.courant() * m.timeStep(h)
	s.mat = m.materials(nz, h)
	s.srcArea = math.Pi * h * h / 4

	s.lossy = m.Relax.TauEps > m.Relax.TauSigma
	if s.lossy {
		tau := m.Relax.TauSigma
		kappa := m.Relax.Delta() / (1 + m.Relax.Delta())
		den := 1 + s.dt/(2*tau)
		s.memDecay = (1 - s.dt/(2*tau)) / den
		s.memDrive = (s.dt * kappa / tau) / den
	}

	n := nr * nz
	for _, f := range []*[]float64{&s.vr, &s.vz, &s.trr, &s.tqq, &s.tzz, &s.trz,
		&s.psiTrrR, &s.psiTrzZ, &s.psiTrzR, &s.psiTzzZ, &s.psiVrR, &s.psiVzZ, &s.psiVzR, &s.psiVrZ} {
		*f = make([]float64, n)
	}
	if s.lossy {
		for _, f := range []*[]float64{&s.rrr, &s.rqq, &s.rzz, &s.rrz} {
			*f = make([]float64, n)
		}
	}

	thickness := float64(pml) * h
	vMax := m.maxVelocity()
	refl := m.pmlReflection()
	dom := s.dominantFreq()
	s.pRint = newProfile(nr, h, 0, float64(m.MaxRange), thickness, vMax, refl, dom, s.dt)
	s.pRhalf = newProfile(nr, h, 0.5, float64(m.MaxRange), thickness, vMax, refl, dom, s.dt)
	s.pZint = newProfile(nz, h, 0, float64(m.Depth), thickness, vMax, refl, dom, s.dt)
	s.pZhalf = newProfile(nz, h, 0.5, float64(m.Depth), thickness, vMax, refl, dom, s.dt)
	return s, nil
}

// dominantFreq is the frequency the C-PML shift is tuned to.
//
// The shift matters most for energy arriving at grazing incidence, and the
// grazing arrival that matters here is the Rayleigh wave. Absent a stated
// frequency, the middle of the band the grid resolves well is the best guess
// available, and the layer is forgiving of being wrong by a factor of two.
func (s *Sim) dominantFreq() float64 {
	if s.m.DominantFreq > 0 {
		return float64(s.m.DominantFreq)
	}
	slowest, _ := s.m.Stack.VelocityBounds()
	v := float64(slowest)
	if s.lossy {
		v = s.m.Relax.RelaxedVelocity(v, s.m.refFreq())
	}
	return v / (20 * s.dr)
}

// Dt is the time step.
func (s *Sim) Dt() float64 { return s.dt }

// Cells is the grid size including the absorbing layer.
func (s *Sim) Cells() (nr, nz int) { return s.nr, s.nz }

// Time is the instant the velocities currently hold, which is half a step
// behind the stresses.
func (s *Sim) Time() float64 { return (float64(s.step) - 0.5) * s.dt }

// Drive sets the vertical force applied at the axis, one sample per time step.
// Steps past the end of the series apply no force.
func (s *Sim) Drive(force []float64) { s.force = force }

func (s *Sim) forceAt(n int) float64 {
	if n < 0 || n >= len(s.force) {
		return 0
	}
	return s.force[n]
}

// Column is the grid column nearest a range, and the range it actually sits
// at. Receivers snap to the grid rather than being interpolated: interpolating
// a receiver would blur the arrival time by exactly the quantity the arrival
// time is being measured to check.
func (s *Sim) Column(r units.Metres) (int, units.Metres) {
	i := int(math.Round(float64(r) / s.dr))
	i = max(0, min(i, s.nr-2))
	return i, units.Metres(float64(i) * s.dr)
}

// Step advances the simulation by one time step.
func (s *Sim) Step() {
	// The applied traction is a prescribed boundary value, so it is written
	// rather than accumulated: the free surface has no equation of motion.
	s.tzz[0] = -s.forceAt(s.step) / s.srcArea
	s.velocities()
	s.stresses((s.forceAt(s.step+1) - s.forceAt(s.step)) / s.dt)
	s.step++
}

func (s *Sim) velocities() {
	nr, nz := s.nr, s.nz
	dr, dz, dt := s.dr, s.dz, s.dt

	// Radial velocity: half a cell out in radius, on a row of nodes.
	for j := range nz - 1 {
		zp := j >= s.pZint.first
		row := j * nr
		rho := s.mat.rho[j]
		for i := range nr - 1 {
			k := row + i
			dtrr := (s.trr[k+1] - s.trr[k]) / dr
			if i >= s.pRhalf.first {
				dtrr += s.pRhalf.update(&s.psiTrrR[k], i, dtrr)
			}
			var dtrz float64
			if j == 0 {
				// The free surface images the shear stress antisymmetrically,
				// so the half cell above the surface carries minus the half
				// cell below and the difference doubles.
				dtrz = 2 * s.trz[k] / dz
			} else {
				dtrz = (s.trz[k] - s.trz[k-nr]) / dz
			}
			if zp {
				dtrz += s.pZint.update(&s.psiTrzZ[k], j, dtrz)
			}
			hoop := 0.5 * ((s.trr[k] - s.tqq[k]) + (s.trr[k+1] - s.tqq[k+1])) / ((float64(i) + 0.5) * dr)
			s.vr[k] += dt / rho * (dtrr + dtrz + hoop)
		}
	}

	// Vertical velocity: half a cell down, on a column of nodes.
	for j := range nz - 1 {
		zh := j >= s.pZhalf.first
		row := j * nr
		rho := s.mat.rhoZ[j]
		for i := range nr - 1 {
			k := row + i
			dtzz := (s.tzz[k+nr] - s.tzz[k]) / dz
			if zh {
				dtzz += s.pZhalf.update(&s.psiTzzZ[k], j, dtzz)
			}
			var radial float64
			if i == 0 {
				// On the axis the shear stress vanishes and its gradient and
				// its ratio to radius take the same limit, so the two terms
				// coincide and the antisymmetric image supplies both.
				radial = 4 * s.trz[k] / dr
			} else {
				d := (s.trz[k] - s.trz[k-1]) / dr
				if i >= s.pRint.first {
					d += s.pRint.update(&s.psiTrzR[k], i, d)
				}
				radial = d + 0.5*(s.trz[k]+s.trz[k-1])/(float64(i)*dr)
			}
			s.vz[k] += dt / rho * (dtzz + radial)
		}
	}
}

func (s *Sim) stresses(dTdt float64) {
	nr, nz := s.nr, s.nz
	dr, dz, dt := s.dr, s.dz, s.dt

	// Normal stresses, at whole nodes.
	for j := range nz {
		zp := j >= s.pZint.first
		row := j * nr
		lam, mu := s.mat.lamU[j], s.mat.muU[j]
		lam2mu := lam + 2*mu
		for i := range nr {
			k := row + i
			var dvr, vrr float64
			if i == 0 {
				// On the axis the radial velocity vanishes and v_r/r tends to
				// its own derivative, so one antisymmetric image gives both.
				dvr = 2 * s.vr[k] / dr
				vrr = dvr
			} else {
				dvr = (s.vr[k] - s.vr[k-1]) / dr
				vrr = 0.5 * (s.vr[k] + s.vr[k-1]) / (float64(i) * dr)
			}
			if i >= s.pRint.first {
				dvr += s.pRint.update(&s.psiVrR[k], i, dvr)
			}
			var dvz float64
			if j == 0 {
				// At the surface the vertical stress is prescribed, so its
				// rate is known and the vertical strain rate follows from it
				// rather than from a difference across a row that is not
				// there. Holding the vertical memory variable at zero is
				// consistent: a prescribed stress drives its own memory
				// variable to zero, which is where it stays.
				applied := 0.0
				if i == 0 {
					applied = dTdt
				}
				dvz = (applied - lam*(dvr+vrr)) / lam2mu
			} else {
				dvz = (s.vz[k] - s.vz[k-nr]) / dz
				if zp {
					dvz += s.pZint.update(&s.psiVzZ[k], j, dvz)
				}
			}
			err := lam2mu*dvr + lam*(vrr+dvz)
			eqq := lam*(dvr+dvz) + lam2mu*vrr
			if s.lossy {
				s.trr[k] += dt * (err + s.relax(&s.rrr[k], err))
				s.tqq[k] += dt * (eqq + s.relax(&s.rqq[k], eqq))
			} else {
				s.trr[k] += dt * err
				s.tqq[k] += dt * eqq
			}
			if j == 0 {
				continue
			}
			ezz := lam*(dvr+vrr) + lam2mu*dvz
			if s.lossy {
				s.tzz[k] += dt * (ezz + s.relax(&s.rzz[k], ezz))
			} else {
				s.tzz[k] += dt * ezz
			}
		}
	}

	// Shear stress, half a cell out and half a cell down.
	for j := range nz - 1 {
		zh := j >= s.pZhalf.first
		row := j * nr
		mu := s.mat.muZ[j]
		for i := range nr - 1 {
			k := row + i
			dvzdr := (s.vz[k+1] - s.vz[k]) / dr
			if i >= s.pRhalf.first {
				dvzdr += s.pRhalf.update(&s.psiVzR[k], i, dvzdr)
			}
			dvrdz := (s.vr[k+nr] - s.vr[k]) / dz
			if zh {
				dvrdz += s.pZhalf.update(&s.psiVrZ[k], j, dvrdz)
			}
			e := mu * (dvzdr + dvrdz)
			if s.lossy {
				s.trz[k] += dt * (e + s.relax(&s.rrz[k], e))
			} else {
				s.trz[k] += dt * e
			}
		}
	}
}

// relax advances one memory variable and returns what to add to the elastic
// stress rate.
//
// Crank-Nicolson rather than explicit Euler. The relaxation time can be short
// compared with the time step at high Q, where an explicit update would be the
// one unstable term in an otherwise stable scheme.
func (s *Sim) relax(r *float64, e float64) float64 {
	next := s.memDecay**r - s.memDrive*e
	avg := 0.5 * (*r + next)
	*r = next
	return avg
}

// SurfaceVelocity is the vertical particle velocity at the free surface in a
// grid column, positive downward.
//
// The vertical velocity lives half a cell below the surface, and simply
// reporting that value carries a bias that does not shrink when the comparison
// tolerance does. Extrapolating with the free-surface condition removes it: the
// vertical strain rate at the surface is fixed by the vanishing of the normal
// stress, so the half-cell correction is known rather than guessed.
func (s *Sim) SurfaceVelocity(i int) units.Velocity {
	var dvr, vrr float64
	if i == 0 {
		dvr = 2 * s.vr[0] / s.dr
		vrr = dvr
	} else {
		dvr = (s.vr[i] - s.vr[i-1]) / s.dr
		vrr = 0.5 * (s.vr[i] + s.vr[i-1]) / (float64(i) * s.dr)
	}
	lam, mu := s.mat.lamU[0], s.mat.muU[0]
	dvz := -lam / (lam + 2*mu) * (dvr + vrr)
	return units.Velocity(s.vz[i] - 0.5*s.dz*dvz)
}

// RadialVelocity is the radial particle velocity at the free surface, half a
// cell out from the column index, positive outward.
func (s *Sim) RadialVelocity(i int) units.Velocity { return units.Velocity(s.vr[i]) }

// MaxSpeed is the largest particle velocity anywhere on the grid. It exists to
// detect divergence, which is what an unstable run does long before it produces
// a wrong answer that looks plausible.
func (s *Sim) MaxSpeed() float64 {
	m := 0.0
	for _, v := range s.vz {
		m = math.Max(m, math.Abs(v))
	}
	for _, v := range s.vr {
		m = math.Max(m, math.Abs(v))
	}
	return m
}
