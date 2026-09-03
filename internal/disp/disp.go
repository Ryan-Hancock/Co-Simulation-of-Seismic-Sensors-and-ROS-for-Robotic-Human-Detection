// Package disp finds the Rayleigh modes of a layered half-space: the phase
// velocities at which the secular function vanishes, as a function of
// frequency.
//
// The plan calls a root finder that misses or jumps modes the worst failure
// mode in the whole work package, "because it looks plausible". A dispersion
// curve that has silently swapped onto a higher mode partway along is smooth,
// monotone and entirely wrong, and everything downstream — travel times,
// waveforms, arrival picks — inherits the error without any sign of it.
//
// So the strategy here is bracketing rather than continuation. Every root is
// found by scanning for a sign change on a grid and bisecting inside it, at
// every frequency independently. Continuation — following a root from the
// previous frequency — is faster and is what tempts a solver into mode
// jumping, because when two modes approach each other it will happily follow
// the wrong one. Scanning cannot jump: it can only fail to see a root, and
// that failure is detectable by counting.
package disp

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/propmat"
	"geosim.dev/geosim/internal/units"
)

// Search configures the root scan.
type Search struct {
	// Samples is how many velocities the scan tries between the bounds.
	// Missing a mode means two roots fell inside one interval, so this is the
	// parameter that trades cost against the risk of the failure that matters.
	// Zero uses 2000.
	Samples int
	// Tolerance is the relative precision of each root. Zero uses 1e-10.
	Tolerance float64
}

func (s Search) samples() int {
	if s.Samples > 0 {
		return s.Samples
	}
	return 2000
}

func (s Search) tolerance() float64 {
	if s.Tolerance > 0 {
		return s.Tolerance
	}
	return 1e-10
}

// Bounds are the velocity range modes can occupy.
//
// No mode travels slower than the Rayleigh velocity of the slowest material,
// which is about 0.87 of its shear velocity at worst; and none travels as fast
// as the half-space's shear velocity, because above that the wave radiates
// downward instead of being trapped. Searching outside either bound wastes
// time at best and finds spurious sign changes at worst.
func Bounds(s layer.Stack) (lo, hi float64) {
	slowest, _ := s.VelocityBounds()
	return 0.85 * float64(slowest), float64(s.HalfSpace().Vs)
}

// Modes returns every Rayleigh phase velocity at the given frequency, slowest
// first: index 0 is the fundamental, 1 the first higher mode, and so on.
func Modes(s layer.Stack, freq float64, opt Search) ([]units.SpeedMPS, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if freq <= 0 {
		return nil, fmt.Errorf("disp: frequency must be positive, got %g Hz", freq)
	}
	lo, hi := Bounds(s)
	n := opt.samples()

	f := func(c float64) (float64, bool) {
		v, err := propmat.SecularAtVelocity(s, freq, c)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, false
		}
		return v, true
	}

	// The scan stops just short of the half-space shear velocity, where the
	// secular function is undefined rather than merely large.
	step := (hi*(1-1e-9) - lo) / float64(n)
	var out []units.SpeedMPS

	prevC := lo
	prevV, prevOK := f(prevC)
	for i := 1; i <= n; i++ {
		c := lo + float64(i)*step
		v, ok := f(c)
		if ok && prevOK && signChange(prevV, v) {
			root, err := bisect(f, prevC, c, opt.tolerance())
			if err == nil {
				out = append(out, units.SpeedMPS(root))
			}
		}
		prevC, prevV, prevOK = c, v, ok
	}
	return out, nil
}

// Mode returns one Rayleigh mode's phase velocity, or an error if the stack
// does not support that many modes at this frequency.
//
// Higher modes have cutoff frequencies below which they simply do not exist,
// so an error here is often the physics rather than a failure — which is why
// it says how many were found.
func Mode(s layer.Stack, freq float64, mode int, opt Search) (units.SpeedMPS, error) {
	if mode < 0 {
		return 0, fmt.Errorf("disp: mode index must not be negative, got %d", mode)
	}
	all, err := Modes(s, freq, opt)
	if err != nil {
		return 0, err
	}
	if mode >= len(all) {
		return 0, fmt.Errorf("disp: mode %d does not exist at %g Hz (found %d)", mode, freq, len(all))
	}
	return all[mode], nil
}

// Curve is one mode's phase velocity across a set of frequencies. Entries
// where the mode does not exist are omitted, so the two slices stay aligned
// with each other rather than with the requested frequencies.
type Curve struct {
	Mode          int
	Frequency     []float64
	PhaseVelocity []units.SpeedMPS
}

// PhaseCurve computes one mode across a frequency list.
func PhaseCurve(s layer.Stack, freqs []float64, mode int, opt Search) (Curve, error) {
	if err := s.Validate(); err != nil {
		return Curve{}, err
	}
	c := Curve{Mode: mode}
	for _, f := range freqs {
		v, err := Mode(s, f, mode, opt)
		if err != nil {
			continue // the mode has not cut in yet at this frequency
		}
		c.Frequency = append(c.Frequency, f)
		c.PhaseVelocity = append(c.PhaseVelocity, v)
	}
	if len(c.Frequency) == 0 {
		return c, fmt.Errorf("disp: mode %d exists at none of the %d frequencies given", mode, len(freqs))
	}
	return c, nil
}

// GroupVelocity is U = d(omega)/dk, computed from the phase curve by central
// difference in frequency:
//
//	U = c / (1 - (f/c) * dc/df)
//
// It is what carries energy, and so what sets when a wave packet arrives —
// the phase velocity only says how the crests move. In a dispersive medium
// they differ enough to matter for arrival picking.
func GroupVelocity(s layer.Stack, freq float64, mode int, opt Search) (units.SpeedMPS, error) {
	const rel = 1e-3
	df := freq * rel
	lo, err := Mode(s, freq-df, mode, opt)
	if err != nil {
		return 0, err
	}
	hi, err := Mode(s, freq+df, mode, opt)
	if err != nil {
		return 0, err
	}
	c, err := Mode(s, freq, mode, opt)
	if err != nil {
		return 0, err
	}
	dcdf := (float64(hi) - float64(lo)) / (2 * df)
	den := 1 - freq/float64(c)*dcdf
	if den == 0 {
		return 0, fmt.Errorf("disp: group velocity is singular at %g Hz", freq)
	}
	return units.SpeedMPS(float64(c) / den), nil
}

func signChange(a, b float64) bool {
	return (a < 0 && b > 0) || (a > 0 && b < 0)
}

// bisect narrows a bracketed sign change. Bisection rather than anything
// faster because the secular function spans many decades and is very steep
// near some roots, where a secant or Newton step would leave the bracket and
// converge on the wrong root — which is the failure this package exists to
// avoid.
func bisect(f func(float64) (float64, bool), lo, hi, tol float64) (float64, error) {
	flo, ok := f(lo)
	if !ok {
		return 0, fmt.Errorf("disp: secular function undefined at %g", lo)
	}
	for range 200 {
		mid := 0.5 * (lo + hi)
		if mid == lo || mid == hi || (hi-lo) < tol*mid {
			return mid, nil
		}
		v, ok := f(mid)
		if !ok {
			return 0, fmt.Errorf("disp: secular function undefined at %g", mid)
		}
		if signChange(flo, v) {
			hi = mid
		} else {
			lo, flo = mid, v
		}
	}
	return 0.5 * (lo + hi), nil
}
