// Package source schedules the forces applied to the ground: which contacts
// are active, where they are, and what they push with.
//
// It is deliberately not footstep-specific. O3 predicts the robot's own
// seismic signature "via the same Green's function machinery" used for a
// human, which only works if the source stage treats a wheel or a track
// segment on the same footing as a foot. So a contact here is a force profile
// applied at a place over an interval, and a walking gait is one Schedule
// among the several this project will need. Retrofitting that generality after
// WP3 had been written against a footstep-shaped API would have meant
// rewriting every consumer.
//
// The split is between *what a contact pushes with*, which is a Profile, and
// *which contacts happen when and where*, which is a Schedule. Those change
// independently: a different shoe changes the profile, a different route
// changes the schedule.
package source

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/units"
)

// Profile is the force one contact applies to the ground over its life,
// in world-frame newtons as (fx, fy, fz), with fz positive downward into the
// ground.
type Profile interface {
	// Duration is how long the contact lasts.
	Duration() units.Seconds
	// ForceAt is the force at time since the contact began. Outside
	// [0, Duration) it must be zero.
	ForceAt(since units.Seconds) [3]float64
}

// Contact is one contact event: a profile, applied somewhere, starting at some
// time.
type Contact struct {
	// ID names the contact — "left_foot", "wheel_fr", "track_l_seg3". Stable
	// across a run so a consumer can keep per-contact filter state.
	ID string
	// X and Y are the contact point on the surface, in metres.
	X, Y float64
	// Start is when the contact begins.
	Start units.Seconds
	// Profile is what it pushes with.
	Profile Profile
}

// End is when the contact's force stops.
func (c Contact) End() units.Seconds { return c.Start + c.Profile.Duration() }

// ForceAt is the force at absolute time t, zero outside the contact.
func (c Contact) ForceAt(t units.Seconds) [3]float64 {
	return c.Profile.ForceAt(t - c.Start)
}

// Schedule produces the contacts that begin within a time window.
//
// Windowed rather than streamed one at a time so a caller can look ahead — the
// engine needs to know which ranges it will be asked for before it is asked,
// so it can have the Green's functions ready.
type Schedule interface {
	// Contacts returns every contact beginning in [from, to), in time order.
	Contacts(from, to units.Seconds) []Contact
}

// Footfall adapts a stance profile to a heading, resolving the fore-aft shear
// into world axes.
//
// The heading matters because horizontal force radiates with a cos(azimuth)
// pattern about the source-to-receiver line: a walker's shear reaches a sensor
// ahead of or behind them and not one directly abeam. A model that dropped the
// heading would be quietly assuming the walker always faces the sensor.
type Footfall struct {
	Stance  grf.Stance
	Heading float64 // direction of travel in the x-y plane, radians from +x
}

func (f Footfall) Duration() units.Seconds { return f.Stance.Duration }

func (f Footfall) ForceAt(since units.Seconds) [3]float64 {
	if since <= 0 || since >= f.Stance.Duration {
		return [3]float64{}
	}
	tau := float64(since) / float64(f.Stance.Duration)
	w := float64(f.Stance.BodyWeight())
	// The fore-aft shear is defined along the direction of travel: negative is
	// braking, which acts backwards on the ground.
	ap := f.Stance.AnteriorPosteriorAt(tau) * w
	return [3]float64{
		ap * math.Cos(f.Heading),
		ap * math.Sin(f.Heading),
		f.Stance.ProfileAt(tau) * w,
	}
}

// Walk is a person walking a straight line at constant speed: two feet,
// alternating, each planted at its own place.
//
// Two feet at separate positions rather than one source at an average
// position, because the separation is the whole point. Successive footfalls
// are most of a metre apart, so as a walker passes a sensor the range to each
// step differs — and that changing range, not the force profile, is what
// produces the rise and fall of a walk-past. Collapsing the feet to one point
// would remove the signature WP4 goes to the field to measure.
type Walk struct {
	// Stance is the per-foot force profile.
	Stance grf.Stance
	// Speed is the walking speed in metres per second.
	Speed float64
	// StartX, StartY is where the first foot lands.
	StartX, StartY float64
	// Heading is the direction of travel, radians from +x.
	Heading float64
	// StrideLength is the distance between successive contacts of the *same*
	// foot; a step is half of it. Zero derives it from speed and the stance
	// duration.
	StrideLength float64
	// Width is the lateral separation between the two feet, about 0.1 to 0.15
	// metres for normal walking. Feet alternate either side of the path.
	Width float64
	// FirstFoot names the foot that lands first; the other alternates with it.
	FirstFoot, SecondFoot string
	// Until is when the walker stops. Zero means they never do.
	//
	// A bounded walk is not a convenience: an engine that pre-builds a Green's
	// function per footfall has to know there are finitely many. Left
	// unbounded, it discovers new footfalls at ever-increasing range forever
	// and builds a new response for each, inside the real-time loop.
	Until units.Seconds
}

// Defaults for a comfortable walk.
const (
	defaultWidth      = 0.12
	defaultFirstFoot  = "left_foot"
	defaultSecondFoot = "right_foot"
)

// Validate reports whether the walk is physically sensible.
func (w Walk) Validate() error {
	switch {
	case w.Speed <= 0:
		return fmt.Errorf("source: walking speed must be positive, got %g m/s", w.Speed)
	case w.Width < 0:
		return fmt.Errorf("source: stance width must not be negative, got %g m", w.Width)
	case w.StrideLength < 0:
		return fmt.Errorf("source: stride length must not be negative, got %g m", w.StrideLength)
	}
	return w.Stance.Validate()
}

// StrideOrDefault is the distance between successive contacts of the same
// foot.
//
// Derived, when not given, from the observation that stance occupies about
// 60% of the gait cycle: the cycle is Duration/0.6, and a stride is one
// cycle's worth of travel. That keeps speed, cadence and stance duration
// mutually consistent instead of letting a config set three numbers that
// cannot all be true.
func (w Walk) StrideOrDefault() float64 {
	if w.StrideLength > 0 {
		return w.StrideLength
	}
	return w.Speed * float64(w.Stance.Duration) / grf.StanceFraction
}

func (w Walk) width() float64 {
	if w.Width > 0 {
		return w.Width
	}
	return defaultWidth
}

func (w Walk) feet() (string, string) {
	a, b := w.FirstFoot, w.SecondFoot
	if a == "" {
		a = defaultFirstFoot
	}
	if b == "" {
		b = defaultSecondFoot
	}
	return a, b
}

// StepPeriod is the interval between successive contacts of opposite feet.
func (w Walk) StepPeriod() units.Seconds {
	return units.Seconds(w.StrideOrDefault() / 2 / w.Speed)
}

// ContactAt returns the nth footfall, counting from zero at StartX, StartY.
func (w Walk) ContactAt(n int) Contact {
	step := w.StrideOrDefault() / 2
	along := float64(n) * step
	// Feet alternate either side of the path centreline.
	side := w.width() / 2
	if n%2 == 1 {
		side = -side
	}
	cos, sin := math.Cos(w.Heading), math.Sin(w.Heading)
	first, second := w.feet()
	id := first
	if n%2 == 1 {
		id = second
	}
	return Contact{
		ID:      id,
		X:       w.StartX + along*cos - side*sin,
		Y:       w.StartY + along*sin + side*cos,
		Start:   units.Seconds(float64(n)) * w.StepPeriod(),
		Profile: Footfall{Stance: w.Stance, Heading: w.Heading},
	}
}

// Contacts returns every footfall beginning in [from, to).
func (w Walk) Contacts(from, to units.Seconds) []Contact {
	if to <= from {
		return nil
	}
	period := float64(w.StepPeriod())
	if period <= 0 {
		return nil
	}
	if w.Until > 0 && from >= w.Until {
		return nil
	}
	if w.Until > 0 && to > w.Until {
		to = w.Until
	}
	lo := int(math.Ceil(float64(from) / period))
	if lo < 0 {
		lo = 0
	}
	hi := int(math.Ceil(float64(to) / period))
	var out []Contact
	for n := lo; n < hi; n++ {
		c := w.ContactAt(n)
		if c.Start >= from && c.Start < to {
			out = append(out, c)
		}
	}
	return out
}

// RangeSpan is the closest and furthest a receiver at (rx, ry) gets from any
// contact beginning in [from, to).
//
// The engine uses it to know which Green's functions it will need before it is
// asked for them, so building one never lands in the middle of a real-time
// chunk.
func RangeSpan(s Schedule, rx, ry float64, from, to units.Seconds) (min, max float64, ok bool) {
	min, max = math.Inf(1), 0
	for _, c := range s.Contacts(from, to) {
		r := math.Hypot(rx-c.X, ry-c.Y)
		min = math.Min(min, r)
		max = math.Max(max, r)
		ok = true
	}
	return min, max, ok
}
