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
	//
	// internal/gait does not fit this one. It solves for it, from the
	// momentum balance below.
	MidstanceValley float64
	// DutyFactor is the share of the gait cycle this foot spends on the
	// ground. Zero uses StanceFraction.
	//
	// It is here because the momentum balance depends on it and on nothing
	// else about the timing: the demanded impulse is body weight times the
	// cycle, and the cycle is the stance divided by this. Holding it at a
	// constant while stance duration varies with speed would put an eight
	// percent error into the balance across the walking range, which is the
	// low-frequency content of the radiated field.
	DutyFactor float64
	// HumpWidth is the Gaussian half-width of each peak, as a fraction of
	// stance. Zero uses the default.
	HumpWidth float64
	// TransientPeak is the height of the heel-strike transient above the
	// smooth loading curve, in body weights. Zero uses the default; set it
	// negative to remove the transient entirely.
	//
	// This is the part of the force that matters most and is constrained
	// least. The smooth double hump is quasi-static and carries almost no
	// energy above a few hertz; the heel-strike transient is the impact of
	// the heel, shoe and limb, it rises in tens of milliseconds, and it is
	// therefore most of what a geophone at range actually sees. Slice 0
	// showed the point plainly: with only the smooth curve, the radiated
	// field was dominated by the taper — an artefact of the model rather
	// than a feature of the gait.
	//
	// It is also the parameter most sensitive to footwear and surface, which
	// makes it a primary domain-randomisation axis for O4 rather than a
	// constant to be pinned.
	TransientPeak float64
	// TransientRise is the time from heel contact to the transient's first
	// peak. Ten to thirty milliseconds in shod walking; shorter on hard
	// surfaces and with stiff heels. Zero uses the default.
	TransientRise units.Seconds
	// APPeak is the magnitude of the anterior-posterior shear, in body
	// weights: braking through the first half of stance, propulsion through
	// the second. Around 0.2 BW. Zero uses the default; negative removes it.
	//
	// It is included because a horizontal surface force excites Rayleigh
	// waves too — with a different radiation pattern and a quarter-cycle
	// phase shift — and dropping it is a simplification usually made without
	// being stated.
	APPeak float64
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
	defaultHumpWidth     = 0.11
	defaultTransientPeak = 0.35
	defaultTransientRise = 0.012
	defaultAPPeak        = 0.20
	// defaultTaperFraction is not a free choice: it is the value at which the
	// reference walker's ImpulseRatio comes out at 1.0000, i.e. at which the
	// profile delivers exactly the momentum gravity demands over a gait cycle.
	// At 0.098 of a 0.62 s stance it is a 61 ms rise for the quasi-static
	// part, which is also physiologically reasonable. The constraint picked
	// the number; the plausibility check only confirms it was allowed to.
	//
	// It moved from 0.088 when the heel-strike transient arrived: the
	// transient carries impulse of its own, so the smooth curve has to give
	// some back. If the source model changes again, the balance test will say
	// so rather than letting the error sit unnoticed.
	defaultTaperFraction = 0.098
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

// dutyFactor is the stance share of the gait cycle.
func (s Stance) dutyFactor() float64 {
	if s.DutyFactor > 0 {
		return s.DutyFactor
	}
	return StanceFraction
}

func (s Stance) taperFraction() float64 {
	if s.TaperFraction > 0 {
		return s.TaperFraction
	}
	return defaultTaperFraction
}

func (s Stance) transientPeak() float64 {
	switch {
	case s.TransientPeak > 0:
		return s.TransientPeak
	case s.TransientPeak < 0:
		return 0
	}
	return defaultTransientPeak
}

func (s Stance) transientRise() float64 {
	if s.TransientRise > 0 {
		return float64(s.TransientRise)
	}
	return defaultTransientRise
}

func (s Stance) apPeak() float64 {
	switch {
	case s.APPeak > 0:
		return s.APPeak
	case s.APPeak < 0:
		return 0
	}
	return defaultAPPeak
}

// transientAt is the heel-strike transient at normalised stance phase tau, in
// body weights.
//
// A damped sinusoid starting at heel contact: the impact ringing of the heel
// pad, shoe and limb against the ground. That shape rather than a single spike
// because the transient in force-plate records overshoots and then dips before
// the loading hump takes over, which a one-sided pulse cannot do — and because
// an oscillation redistributes force in time without adding much impulse,
// which keeps it from fighting the momentum constraint the smooth curve was
// fitted under.
func (s Stance) transientAt(tau float64) float64 {
	a := s.transientPeak()
	if a == 0 || tau <= 0 {
		return 0
	}
	// Damping pulls the crest earlier than the quarter period, to 0.2009 of a
	// cycle, so the period is set from that rather than from a quarter — which
	// makes TransientRise the time to peak exactly, the quantity force-plate
	// papers report. The ringing decays over half a cycle: impact transients
	// are visibly gone within about fifty milliseconds, which a decay equal to
	// the period would not manage.
	period := s.transientRise() / transientCrestPhase / float64(s.Duration)
	decay := period / 2
	return a / transientCrest * math.Exp(-tau/decay) * math.Sin(2*math.Pi*tau/period)
}

// The crest of exp(-2u)*sin(2*pi*u) and the phase it occurs at. Dividing by
// the crest makes TransientPeak the height the transient actually reaches
// rather than the amplitude of the sinusoid inside it; setting the period from
// the phase makes TransientRise the time to that peak. Both exist so the two
// parameters mean what a published force trace would call them.
const (
	transientCrest      = 0.63790
	transientCrestPhase = 0.20090
)

// AnteriorPosteriorAt is the fore-aft shear at normalised stance phase tau, in
// body weights: negative (braking) through the first half of stance, positive
// (propulsion) through the second.
//
// A single sine over the stance, which is both a fair approximation of the
// measured shape and exactly impulse-free over the cycle — as it must be, since
// a walker holding a steady speed changes no fore-aft momentum from step to
// step. The taper perturbs that slightly, and the test bounds how much.
func (s Stance) AnteriorPosteriorAt(tau float64) float64 {
	a := s.apPeak()
	if a == 0 || tau <= 0 || tau >= 1 {
		return 0
	}
	return tukey(tau, s.taperFraction()) * -a * math.Sin(2*math.Pi*tau)
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
	// The taper applies to the smooth curve only. It exists to keep the
	// quasi-static force from starting at a non-zero value, and it is about
	// fifty-five milliseconds long — three times the transient's rise. Taking
	// the transient through it as well suppressed its peak fourfold, which
	// would have quietly removed most of the radiated signal while leaving a
	// force trace that still looked like a footstep. The transient needs no
	// taper of its own: a damped sinusoid starting at heel contact is already
	// zero there, and its fast rise from zero *is* the physical heel strike.
	return tukey(tau, s.taperFraction())*body + s.transientAt(tau)
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
	cycle := float64(s.Duration) / s.dutyFactor()
	demanded := float64(s.BodyWeight()) * cycle
	return 2 * s.Impulse() / demanded
}

// BalancedValley is the midstance valley at which the profile delivers exactly
// the momentum gravity demands, given everything else about the stance.
//
// The valley is solved for rather than fitted because the momentum balance is
// the one hard constraint on a profile whose shape is otherwise read off
// published figures, and because the valley is the part of the M those figures
// pin down least — the two peaks are what a force plate reports most reliably
// and what the literature tabulates against speed. Spending the constraint on
// the least certain parameter is the right way round.
//
// It also makes the balance hold across the whole walking range instead of at
// one gait. The demanded impulse scales as one over the duty factor, which
// falls from about 0.65 at a slow walk to 0.59 at a brisk one, so a profile
// tuned at a single speed is eight percent out at the other end. That error is
// entirely in the low-frequency content of the radiated field, where no amount
// of correct propagation modelling can recover it.
//
// The impulse rises monotonically with the valley, so a bisection cannot land
// on the wrong root; there is only one.
func (s Stance) BalancedValley() (float64, error) {
	f := func(v float64) float64 {
		t := s
		t.MidstanceValley = v
		return t.ImpulseRatio() - 1
	}
	lo, hi := 0.05, 1.5
	flo, fhi := f(lo), f(hi)
	if flo > 0 || fhi < 0 {
		return 0, fmt.Errorf("grf: no valley between %g and %g body weights balances the momentum "+
			"of a %g s stance at duty %g with peaks %g and %g",
			lo, hi, float64(s.Duration), s.dutyFactor(), s.FirstPeak, s.SecondPeak)
	}
	for range 60 {
		mid := 0.5 * (lo + hi)
		if f(mid) < 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	return 0.5 * (lo + hi), nil
}

// PeakForce is the largest vertical force during the stance.
//
// Scanned rather than taken as the larger of the two hump parameters: the
// heel-strike transient rings through the loading phase and lifts the first
// peak slightly above its nominal value, so the two are no longer the same
// number.
func (s Stance) PeakForce() units.Newtons {
	const steps = 20000
	var peak float64
	for i := range steps {
		peak = math.Max(peak, s.ProfileAt((float64(i)+0.5)/steps))
	}
	return units.Newtons(peak * float64(s.BodyWeight()))
}
