// Package grf is the seismic source: the vertical ground reaction force a
// footstep applies to the ground.
//
// This is the weakest link in the forward model and the plan says so. Isaac's
// animated characters are kinematic, so their contact forces are not physical
// and cannot be used directly; what stands in their place is a parametric
// profile driven by mass, walking speed and gait phase, and validated against
// force-plate literature rather than derived from first principles. It is a
// stated modelling assumption with a sensitivity analysis attached, not a
// measurement.
//
// Slice 0 models the smooth part only: the double hump of a stance phase. The
// heel-strike transient, the anterior-posterior component, and the scheduling
// of two feet into a gait arrive in slice 2. That ordering matters, because
// the transient is where the high-frequency energy lives and therefore most of
// what makes a footstep detectable at range — so slice 0's waveform is
// expected to be sluggish, and that is the point it is making.
package grf

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/units"
)

// Stance is one foot's contact with the ground, as a vertical force profile.
//
// The shape is the classic M: a first peak as weight is accepted onto the
// limb, a midstance valley as the body vaults over the foot, and a second peak
// as the ankle pushes off. Peak values are quoted in multiples of body weight,
// which is how the biomechanics literature reports them and how they stay
// comparable across subjects.
type Stance struct {
	// Mass is the subject's body mass.
	Mass units.Kilograms
	// Duration is the stance phase — how long this foot is on the ground.
	// Roughly 0.6 to 0.7 s at comfortable walking speed, shortening as speed
	// rises.
	Duration units.Seconds
	// FirstPeak and SecondPeak are the loading-response and push-off maxima,
	// in multiples of body weight. Typically 1.1 to 1.3, rising with speed.
	FirstPeak, SecondPeak float64
	// MidstanceValley is the trough between them, again in body weights.
	// Typically 0.7 to 0.8; it deepens as walking speed rises because the
	// body's centre of mass is accelerating upward over the stance foot.
	MidstanceValley float64
	// HumpWidth is the Gaussian half-width of each peak, as a fraction of
	// stance. Zero uses the default.
	HumpWidth float64
	// TaperFraction is the fraction of stance at each end over which the
	// force rises from and returns to zero. Zero uses the default.
	//
	// This is not cosmetic. A profile that starts at a non-zero value is a
	// step in the force, and a step radiates like an impulse — its spectrum
	// falls off only as 1/f and would swamp the real signal at exactly the
	// frequencies the detector cares about. The force must reach zero
	// continuously at heel strike and toe-off, and this is what makes it.
	TaperFraction float64
}

// Default profile constants, chosen so that the impulse over a gait cycle
// comes out right — see ImpulseRatio.
const (
	defaultHumpWidth = 0.11
	// defaultTaperFraction is not a free choice: it is the value at which the
	// reference walker's ImpulseRatio comes out at 1.00004, i.e. at which the
	// profile delivers exactly the momentum gravity demands over a gait cycle.
	// At 0.088 of a 0.62 s stance it is a 55 ms rise, which is also a
	// physically reasonable loading time for the smooth part of the force.
	// The constraint picked the number; the plausibility check only confirms
	// it was allowed to.
	defaultTaperFraction = 0.088
	firstPeakPhase       = 0.25
	secondPeakPhase      = 0.75
	// StanceFraction is the share of the gait cycle one foot spends on the
	// ground in normal walking. The remainder is swing; the overlap between
	// the two feet is double support.
	StanceFraction = 0.60
)

// Walker is a comfortable walking stance for a subject of the given mass:
// 75 kg at roughly 1.3 m/s if unspecified.
func Walker(mass units.Kilograms) Stance {
	return Stance{
		Mass:            mass,
		Duration:        0.62,
		FirstPeak:       1.15,
		SecondPeak:      1.12,
		MidstanceValley: 0.75,
	}
}

// Validate reports whether the profile is physically sensible.
func (s Stance) Validate() error {
	switch {
	case s.Mass <= 0:
		return fmt.Errorf("grf: mass must be positive, got %g kg", s.Mass)
	case s.Duration <= 0:
		return fmt.Errorf("grf: stance duration must be positive, got %g s", s.Duration)
	case s.FirstPeak <= 0 || s.SecondPeak <= 0:
		return fmt.Errorf("grf: peaks must be positive, got %g and %g BW", s.FirstPeak, s.SecondPeak)
	case s.MidstanceValley <= 0:
		return fmt.Errorf("grf: midstance valley must be positive, got %g BW", s.MidstanceValley)
	case s.MidstanceValley >= s.FirstPeak || s.MidstanceValley >= s.SecondPeak:
		return fmt.Errorf("grf: valley %g BW must lie below both peaks (%g, %g BW), or the profile is not a double hump",
			s.MidstanceValley, s.FirstPeak, s.SecondPeak)
	case s.HumpWidth < 0 || s.HumpWidth >= 0.5:
		return fmt.Errorf("grf: hump width must be in [0, 0.5) of stance, got %g", s.HumpWidth)
	case s.TaperFraction < 0 || s.TaperFraction > 0.5:
		return fmt.Errorf("grf: taper fraction must be in [0, 0.5], got %g", s.TaperFraction)
	}
	return nil
}

// BodyWeight is the static weight of the subject.
func (s Stance) BodyWeight() units.Newtons {
	return units.Newtons(float64(s.Mass) * units.GravityMPS2)
}

func (s Stance) humpWidth() float64 {
	if s.HumpWidth > 0 {
		return s.HumpWidth
	}
	return defaultHumpWidth
}

func (s Stance) taperFraction() float64 {
	if s.TaperFraction > 0 {
		return s.TaperFraction
	}
	return defaultTaperFraction
}

// ProfileAt is the vertical force in multiples of body weight at normalised
// stance phase tau, which runs 0 at heel strike to 1 at toe-off. Outside that
// interval the foot is in the air and the force is zero.
//
// Built as a tapered support at the valley level with two Gaussian humps on
// top. Parameterising it by the three quantities force plates actually report
// — two peaks and a valley — rather than by Gaussian amplitudes means the
// numbers in a config file can be read straight off a published figure.
func (s Stance) ProfileAt(tau float64) float64 {
	if tau <= 0 || tau >= 1 {
		return 0
	}
	w := s.humpWidth()
	g1 := math.Exp(-math.Pow((tau-firstPeakPhase)/w, 2))
	g2 := math.Exp(-math.Pow((tau-secondPeakPhase)/w, 2))
	body := s.MidstanceValley +
		(s.FirstPeak-s.MidstanceValley)*g1 +
		(s.SecondPeak-s.MidstanceValley)*g2
	return tukey(tau, s.taperFraction()) * body
}

// tukey is a raised-cosine taper: zero at the ends, unity across the interior,
// with a cosine transition of width r at each end.
func tukey(tau, r float64) float64 {
	switch {
	case r <= 0:
		return 1
	case tau < r:
		return 0.5 * (1 - math.Cos(math.Pi*tau/r))
	case tau > 1-r:
		return 0.5 * (1 - math.Cos(math.Pi*(1-tau)/r))
	default:
		return 1
	}
}

// ForceAt is the vertical ground reaction force at time t after heel strike.
func (s Stance) ForceAt(t units.Seconds) units.Newtons {
	return units.Newtons(s.ProfileAt(float64(t)/float64(s.Duration)) * float64(s.BodyWeight()))
}

// Sample renders the stance as a force time series at sample rate fs, with
// lead and tail seconds of silence around it. The returned slice is in newtons
// and starts at -lead relative to heel strike.
func (s Stance) Sample(fs float64, lead, tail units.Seconds) ([]units.Newtons, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if fs <= 0 {
		return nil, fmt.Errorf("grf: sample rate must be positive, got %g", fs)
	}
	if lead < 0 || tail < 0 {
		return nil, fmt.Errorf("grf: lead and tail must not be negative, got %g and %g s", lead, tail)
	}
	total := float64(lead) + float64(s.Duration) + float64(tail)
	n := int(math.Round(total * fs))
	out := make([]units.Newtons, n)
	for i := range out {
		out[i] = s.ForceAt(units.Seconds(float64(i)/fs - float64(lead)))
	}
	return out, nil
}

// Impulse is the integral of the vertical force over the stance, in newton
// seconds.
func (s Stance) Impulse() float64 {
	const steps = 20000
	var sum float64
	for i := range steps {
		sum += s.ProfileAt((float64(i) + 0.5) / steps)
	}
	return sum / steps * float64(s.Duration) * float64(s.BodyWeight())
}

// ImpulseRatio is the vertical impulse two feet deliver over one gait cycle,
// divided by the impulse gravity demands over the same cycle.
//
// It must be 1. Over a full cycle in steady walking the body's centre of mass
// returns to the same height with the same velocity, so the ground's total
// vertical impulse has to equal the weight impulse exactly — this is Newton's
// second law integrated over a period, and no amount of parametric
// curve-shaping is allowed to violate it.
//
// It is the strongest physical constraint available on a profile whose shape
// is otherwise fitted rather than derived, and it constrains the free
// parameters: it is why the default hump width and taper are what they are.
// A profile that looks right and fails this is delivering the wrong momentum
// to the ground, and will get the low-frequency content of the radiated field
// wrong in a way no amount of correct propagation modelling can recover.
func (s Stance) ImpulseRatio() float64 {
	cycle := float64(s.Duration) / StanceFraction
	demanded := float64(s.BodyWeight()) * cycle
	return 2 * s.Impulse() / demanded
}

// PeakForce is the largest vertical force during the stance.
func (s Stance) PeakForce() units.Newtons {
	return units.Newtons(math.Max(s.FirstPeak, s.SecondPeak) * float64(s.BodyWeight()))
}
