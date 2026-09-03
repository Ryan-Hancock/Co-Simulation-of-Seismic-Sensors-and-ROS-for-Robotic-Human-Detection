package fdtd

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/units"
)

// Ricker is the standard band-limited wavelet: the second derivative of a
// Gaussian, with its amplitude spectrum peaking at f0.
//
// Two properties make it the right drive for this comparison. It has no
// spectral content at zero frequency, so the medium is left with no permanent
// displacement to argue about; and its net impulse is zero, so the half-space
// keeps its momentum and there is no slow drift underneath the arrivals. It is
// delayed by 1.5/f0, far enough that the truncation at t = 0 is at the 1e-9
// level rather than the 1e-3 level a one-period delay would leave.
func Ricker(peak units.Newtons, f0 units.Hertz, dt float64, n int) ([]float64, error) {
	if f0 <= 0 || dt <= 0 || n <= 0 {
		return nil, fmt.Errorf("fdtd: Ricker needs a positive frequency, step and length")
	}
	out := make([]float64, n)
	t0 := 1.5 / float64(f0)
	a := math.Pi * math.Pi * float64(f0) * float64(f0)
	for i := range out {
		tau := float64(i)*dt - t0
		out[i] = float64(peak) * (1 - 2*a*tau*tau) * math.Exp(-a*tau*tau)
	}
	return out, nil
}

// RickerSpectrum is the wavelet's transform, so a frequency-domain reference
// can be driven by exactly the same force without a round trip through an FFT
// of a truncated series.
//
// The convention matches the time series above, delay included: a comparison
// that got the delay wrong would show as a pure phase ramp, which is precisely
// the error a time-domain and a frequency-domain path are most likely to
// disagree by and least likely to notice.
func RickerSpectrum(peak units.Newtons, f0 units.Hertz, f float64) complex128 {
	fr := float64(f) / float64(f0)
	mag := 2 / math.Sqrt(math.Pi) * fr * fr * math.Exp(-fr*fr) / float64(f0)
	phase := -2 * math.Pi * f * 1.5 / float64(f0)
	return complex(float64(peak)*mag, 0) * complex(math.Cos(phase), math.Sin(phase))
}

// Shot is one run of the solver.
type Shot struct {
	Model Model
	// Force is the vertical force at the axis, one sample per time step. Build
	// it with Ricker at the step New reports.
	Force []float64
	// Ranges are where to record. They snap to grid columns.
	Ranges []units.Metres
	// Steps is how long to run. It must outlast the slowest arrival at the
	// furthest receiver, plus the wavelet.
	Steps int
	// Progress, if set, is called every few hundred steps.
	Progress func(step, total int)
}

// Trace is one receiver's record.
type Trace struct {
	// Range is where the receiver actually sits, after snapping to the grid.
	Range units.Metres
	// Vertical is the surface particle velocity, positive downward.
	Vertical []units.Velocity
}

// Result is what a shot recorded.
type Result struct {
	// Dt is the time step.
	Dt float64
	// T0 is the instant of the first sample. Velocities live on half steps,
	// so it is half a step, not zero — and a comparison that assumed zero
	// would carry a fixed timing bias of exactly that size.
	T0     float64
	Traces []Trace
	// Cells is the grid size including the absorbing layer.
	Cells [2]int
	// Peak is the largest particle velocity seen anywhere, at any time. A
	// stable run's peak is set by the source; a diverging one's is not.
	Peak float64
}

// Run drives a shot to completion.
func Run(sh Shot) (*Result, error) {
	s, err := New(sh.Model)
	if err != nil {
		return nil, err
	}
	if sh.Steps <= 0 {
		return nil, fmt.Errorf("fdtd: a shot needs a positive number of steps")
	}
	s.Drive(sh.Force)

	cols := make([]int, len(sh.Ranges))
	res := &Result{Dt: s.Dt(), T0: 0.5 * s.Dt(), Traces: make([]Trace, len(sh.Ranges))}
	res.Cells[0], res.Cells[1] = s.Cells()
	for i, r := range sh.Ranges {
		c, snapped := s.Column(r)
		if snapped >= sh.Model.MaxRange {
			return nil, fmt.Errorf("fdtd: receiver at %g m is outside the interior (%g m)", r, sh.Model.MaxRange)
		}
		cols[i] = c
		res.Traces[i] = Trace{Range: snapped, Vertical: make([]units.Velocity, sh.Steps)}
	}

	for n := range sh.Steps {
		s.Step()
		for i, c := range cols {
			res.Traces[i].Vertical[n] = s.SurfaceVelocity(c)
		}
		if sh.Progress != nil && n%500 == 0 {
			sh.Progress(n, sh.Steps)
		}
		if n%256 == 0 {
			p := s.MaxSpeed()
			res.Peak = math.Max(res.Peak, p)
			if math.IsNaN(p) || math.IsInf(p, 0) {
				return nil, fmt.Errorf("fdtd: diverged at step %d of %d", n, sh.Steps)
			}
		}
	}
	return res, nil
}
