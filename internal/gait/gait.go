// Package gait turns a walking speed into a gait.
//
// It exists because of a measurement. Slice 6's Sobol decomposition gave
// walking speed a total-effect index of essentially zero on every statistic —
// and that was a fact about the model, not about walking. Speed was moving the
// stride geometry and nothing else: cadence, stance duration and the peak
// forces all stayed put, so a walker at 1.8 m/s applied the same force for the
// same time as one at 0.8 m/s and simply covered more ground between steps.
// Read literally, the analysis would have told O4 not to randomise walking
// speed, which is the opposite of the truth.
//
// The relations here are read off the biomechanics literature rather than
// derived, in the same spirit as the force profile itself: a stated modelling
// assumption with a sensitivity analysis attached. What is *not* fitted is the
// midstance valley — that one is solved from the momentum balance, because the
// balance has to hold at every speed and the valley is the parameter published
// figures pin down least.
//
// The heel-strike transient is coupled too, and reluctantly. It has to be: the
// radiated signal at geophone frequencies is mostly the transient, so leaving
// it speed-independent leaves walking speed with a two percent effect on peak
// amplitude — which is the conclusion this package exists to overturn, arrived
// at by a different route. The impact peak does rise with heel contact
// velocity and therefore with walking speed, but the slope below is a plausible
// linear fit rather than a published relation, and it is the least constrained
// number in this package. WP4's force plates should replace it. Its rise time
// is left alone, because that is set by the stiffness of the shoe and the
// ground rather than by how fast the walker is going.
//
// Footwear and surface remain an independent axis, as a multiplier on the
// speed-derived transient rather than as a competing absolute. Two parameters
// that both set the same quantity cancel each other in a variance
// decomposition, which is exactly how walking speed came to look irrelevant.
//
// Inter-subject variation at a fixed speed is not represented at all — two
// people walking at 1.3 m/s here have identical gaits, which they do not in
// life. That is a WP4 item too.
package gait

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/units"
)

// Gait is the timing and geometry a walking speed implies.
type Gait struct {
	Speed units.SpeedMPS
	// StepFrequency is steps per second — twice the stride frequency.
	StepFrequency float64
	// StepLength is the distance between successive contacts of *opposite*
	// feet; StrideLength is between successive contacts of the same foot.
	StepLength, StrideLength units.Metres
	// DutyFactor is the share of the gait cycle one foot is on the ground.
	DutyFactor float64
	// CycleTime is one full stride; StanceDuration is one foot's contact.
	CycleTime, StanceDuration units.Seconds
	// DoubleSupport is how long both feet are down together, per occurrence.
	// It vanishes at the walk-run transition, which is where this model stops
	// applying.
	DoubleSupport units.Seconds
}

// The walk ratio: step length divided by cadence is close to constant across
// walking speeds for a given adult, at about 0.0065 m per step per minute
// (Sekiya and Nagasaki; Egerton and colleagues put healthy adults at 0.0058 to
// 0.0065). It is the cleanest relation in the walking literature and it makes
// cadence and step length each scale as the square root of speed, which is the
// commonly quoted result.
//
// Combined with speed = step length times step frequency, it fixes both from
// speed alone with nothing left over.
const walkRatio = 0.0065

// Duty factor falls with speed: about 0.65 at a slow walk, 0.62 at a
// comfortable one, 0.59 at a brisk one. Below 0.5 the feet are never both down
// and the gait is a run, which this model does not describe.
const (
	dutyAtReference = 0.62
	dutySlope       = -0.06 // per m/s about the reference speed
	referenceSpeed  = 1.3
)

// Peak forces rise with speed and the valley deepens. These are the loading and
// push-off maxima in body weights, from force-plate studies across walking
// speeds — around 1.0 and 1.3 body weights at the slow and fast ends. The
// valley is not here: it is solved for.
const (
	firstPeakAtReference  = 1.15
	firstPeakSlope        = 0.30
	secondPeakAtReference = 1.12
	secondPeakSlope       = 0.26
	apPeakAtReference     = 0.20
	apPeakSlope           = 0.13
	// The heel-strike transient at the reference speed, and its fractional
	// rise per m/s. This slope is the weakest assumption in the package.
	transientAtReference = 0.35
	transientSlope       = 0.50
)

// Speed range over which these relations are asserted. Outside it they are
// extrapolation, and the far side of the fast end is a different gait entirely.
const (
	MinSpeed units.SpeedMPS = 0.5
	MaxSpeed units.SpeedMPS = 2.2
)

// At returns the gait timing and geometry for a walking speed.
func At(speed units.SpeedMPS) (Gait, error) {
	if speed < MinSpeed || speed > MaxSpeed {
		return Gait{}, fmt.Errorf("gait: speed %g m/s is outside the %g to %g m/s range these "+
			"relations were read from; beyond the fast end walking becomes running",
			speed, MinSpeed, MaxSpeed)
	}
	v := float64(speed)
	// cadence C in steps per minute: stepLength = walkRatio*C and
	// v = stepLength*C/60, so C = sqrt(60 v / walkRatio).
	stepsPerSecond := math.Sqrt(60*v/walkRatio) / 60
	stepLength := v / stepsPerSecond

	duty := dutyAtReference + dutySlope*(v-referenceSpeed)
	cycle := 2 / stepsPerSecond
	g := Gait{
		Speed:          speed,
		StepFrequency:  stepsPerSecond,
		StepLength:     units.Metres(stepLength),
		StrideLength:   units.Metres(2 * stepLength),
		DutyFactor:     duty,
		CycleTime:      units.Seconds(cycle),
		StanceDuration: units.Seconds(duty * cycle),
		DoubleSupport:  units.Seconds((2*duty - 1) * cycle / 2),
	}
	if g.DoubleSupport <= 0 {
		return Gait{}, fmt.Errorf("gait: duty factor %.3f at %g m/s leaves no double support; "+
			"that is a run, not a walk", duty, speed)
	}
	return g, nil
}

// Stance is the force profile this gait implies for a subject of the given
// mass, with the midstance valley solved from the momentum balance.
//
// Everything the caller has already decided is passed in through base — mass,
// the heel-strike transient, the hump width — and only the speed-dependent
// quantities are overwritten. That way a config can vary footwear without
// having its gait recomputed underneath it.
func (g Gait) Stance(base grf.Stance) (grf.Stance, error) {
	v := float64(g.Speed)
	s := base
	s.Duration = g.StanceDuration
	s.DutyFactor = g.DutyFactor
	s.FirstPeak = firstPeakAtReference + firstPeakSlope*(v-referenceSpeed)
	s.SecondPeak = secondPeakAtReference + secondPeakSlope*(v-referenceSpeed)
	if base.APPeak == 0 {
		s.APPeak = apPeakAtReference + apPeakSlope*(v-referenceSpeed)
	}
	if base.TransientPeak == 0 {
		s.TransientPeak = transientAtReference * (1 + transientSlope*(v-referenceSpeed))
	}
	valley, err := s.BalancedValley()
	if err != nil {
		return grf.Stance{}, fmt.Errorf("gait: at %g m/s: %w", g.Speed, err)
	}
	s.MidstanceValley = valley
	if err := s.Validate(); err != nil {
		return grf.Stance{}, fmt.Errorf("gait: at %g m/s: %w", g.Speed, err)
	}
	return s, nil
}
