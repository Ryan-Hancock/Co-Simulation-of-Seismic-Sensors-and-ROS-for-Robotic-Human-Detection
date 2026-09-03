package source

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/units"
)

func walker() Walk {
	return Walk{Stance: grf.Walker(75), Speed: 1.3, StartX: -8, StartY: 3, Heading: 0}
}

// Speed, cadence and stance duration cannot be set independently: a config
// that fixes all three is over-determined and at most one combination is
// physical. The stride is derived so they stay consistent.
func TestSpeedCadenceAndStanceAreConsistent(t *testing.T) {
	w := walker()
	cycle := float64(w.Stance.Duration) / grf.StanceFraction
	if got, want := w.StrideOrDefault(), w.Speed*cycle; math.Abs(got-want) > 1e-12 {
		t.Errorf("stride %g m, want speed times gait cycle %g m", got, want)
	}
	// Two steps per cycle, so the walker advances one stride per cycle.
	if got, want := 2*float64(w.StepPeriod()), cycle; math.Abs(got-want) > 1e-12 {
		t.Errorf("two step periods make %g s, want one gait cycle %g s", got, want)
	}
	// And the distance covered per unit time really is the speed.
	c0, c100 := w.ContactAt(0), w.ContactAt(100)
	dist := math.Hypot(c100.X-c0.X, c100.Y-c0.Y)
	if got := dist / float64(c100.Start-c0.Start); math.Abs(got-w.Speed)/w.Speed > 1e-9 {
		t.Errorf("footfalls advance at %g m/s, want %g", got, w.Speed)
	}
}

// Stance lasts longer than a step period, so both feet are on the ground for
// part of every cycle. Double support is not an incidental detail: during it
// two sources radiate at once from different places, which is the interference
// that makes a footstep train look the way it does.
func TestStancesOverlapInDoubleSupport(t *testing.T) {
	w := walker()
	a, b := w.ContactAt(0), w.ContactAt(1)
	if a.End() <= b.Start {
		t.Fatalf("first foot lifts at %g s before the second lands at %g s: no double support", a.End(), b.Start)
	}
	overlap := float64(a.End() - b.Start)
	cycle := 2 * float64(w.StepPeriod())
	// Roughly a tenth of the cycle at each transition, twice per cycle.
	if frac := overlap / cycle; frac < 0.05 || frac > 0.25 {
		t.Errorf("double support is %.1f%% of the cycle, want roughly 10%%", frac*100)
	}
}

// The feet must land in different places, alternating either side of the path.
// Collapsing them onto one point would remove the changing range that produces
// a walk-past signature.
func TestFeetAlternateEitherSideOfThePath(t *testing.T) {
	w := walker()
	w.Heading = 0 // travelling along +x, so the offset is in y
	var sides []float64
	for n := range 6 {
		c := w.ContactAt(n)
		sides = append(sides, c.Y-w.StartY)
	}
	for n, s := range sides {
		want := w.width() / 2
		if n%2 == 1 {
			want = -want
		}
		if math.Abs(s-want) > 1e-12 {
			t.Errorf("footfall %d is %g m off the centreline, want %g", n, s, want)
		}
	}
	// Named consistently, so a consumer can keep per-contact state.
	first, second := w.feet()
	for n := range 6 {
		want := first
		if n%2 == 1 {
			want = second
		}
		if got := w.ContactAt(n).ID; got != want {
			t.Errorf("footfall %d is %q, want %q", n, got, want)
		}
	}
}

// The heading has to rotate the fore-aft shear into world axes, or the model
// is quietly assuming the walker always faces the sensor. Vertical force is
// unaffected by heading; horizontal force follows it exactly.
func TestHeadingOrientsTheShearButNotTheWeight(t *testing.T) {
	stance := grf.Walker(75)
	const tau = 0.25 // braking phase
	at := units.Seconds(tau * float64(stance.Duration))

	east := Footfall{Stance: stance, Heading: 0}.ForceAt(at)
	north := Footfall{Stance: stance, Heading: math.Pi / 2}.ForceAt(at)

	if math.Abs(east[2]-north[2]) > 1e-12 {
		t.Errorf("vertical force changed with heading: %g then %g", east[2], north[2])
	}
	mag := math.Hypot(east[0], east[1])
	if mag < 1 {
		t.Fatalf("no horizontal force to orient: %g N", mag)
	}
	if math.Abs(math.Hypot(north[0], north[1])-mag) > 1e-9 {
		t.Error("horizontal magnitude changed with heading")
	}
	// Travelling east, the shear is along x; travelling north, along y.
	if math.Abs(east[1]) > 1e-9 || math.Abs(north[0]) > 1e-9 {
		t.Errorf("shear not aligned with the heading: east %v, north %v", east, north)
	}
	// And braking acts backwards along the direction of travel.
	if east[0] >= 0 {
		t.Errorf("fore-aft force at the braking phase is %g N, want negative", east[0])
	}
}

func TestForceIsZeroOutsideContact(t *testing.T) {
	f := Footfall{Stance: grf.Walker(75), Heading: 0.7}
	for _, at := range []units.Seconds{-1, -1e-9, 0, f.Duration(), f.Duration() + 1} {
		if got := f.ForceAt(at); got != [3]float64{} {
			t.Errorf("force at %g s is %v, want zero outside contact", at, got)
		}
	}
	c := Contact{Start: 5, Profile: f}
	if got := c.ForceAt(4.9); got != [3]float64{} {
		t.Errorf("force before the contact starts is %v, want zero", got)
	}
	if got := c.ForceAt(5.2); got == [3]float64{} {
		t.Error("force during the contact is zero")
	}
	if want := units.Seconds(5) + f.Duration(); c.End() != want {
		t.Errorf("contact ends at %g s, want %g", c.End(), want)
	}
}

// The window must be half-open and gapless: consecutive windows together must
// yield exactly the contacts one big window does, with none dropped or
// repeated at the seam. The engine advances chunk by chunk, so a contact lost
// at a boundary would be a footstep that silently never happened.
func TestContactWindowsTileWithoutGapsOrRepeats(t *testing.T) {
	w := walker()
	const span = 10.0

	whole := w.Contacts(0, span)
	if len(whole) < 10 {
		t.Fatalf("only %d contacts in %g s; the test is not exercising anything", len(whole), span)
	}

	var pieces []Contact
	const step = 0.01 // finer than a step period, so most windows are empty
	for t0 := 0.0; t0 < span; t0 += step {
		pieces = append(pieces, w.Contacts(units.Seconds(t0), units.Seconds(t0+step))...)
	}
	if len(pieces) != len(whole) {
		t.Fatalf("windowed sweep found %d contacts, one window found %d", len(pieces), len(whole))
	}
	for i := range whole {
		if pieces[i].ID != whole[i].ID || pieces[i].Start != whole[i].Start {
			t.Fatalf("contact %d differs: %v vs %v", i, pieces[i], whole[i])
		}
	}
	// Ordered in time, since the engine relies on it.
	for i := 1; i < len(whole); i++ {
		if whole[i].Start < whole[i-1].Start {
			t.Errorf("contacts out of order at %d: %g after %g", i, whole[i].Start, whole[i-1].Start)
		}
	}
	if got := w.Contacts(5, 5); got != nil {
		t.Errorf("empty window returned %d contacts", len(got))
	}
	if got := w.Contacts(5, 4); got != nil {
		t.Errorf("reversed window returned %d contacts", len(got))
	}
}

// A walk past a sensor must close and then open again, which is the signature
// WP4 goes to the field to measure.
func TestWalkPastRangeClosesThenOpens(t *testing.T) {
	w := walker()
	w.StartX, w.StartY, w.Heading = -10, 4, 0 // passes a sensor at the origin, 4 m abeam

	var ranges []float64
	for _, c := range w.Contacts(0, 16) {
		ranges = append(ranges, math.Hypot(c.X, c.Y))
	}
	if len(ranges) < 8 {
		t.Fatalf("only %d footfalls", len(ranges))
	}
	minAt, minVal := 0, math.Inf(1)
	for i, r := range ranges {
		if r < minVal {
			minVal, minAt = r, i
		}
	}
	if minAt == 0 || minAt == len(ranges)-1 {
		t.Errorf("closest approach at footfall %d of %d: the walk does not pass the sensor", minAt, len(ranges))
	}
	if math.Abs(minVal-4) > 0.5 {
		t.Errorf("closest approach %.2f m, want about the 4 m offset", minVal)
	}

	lo, hi, ok := RangeSpan(w, 0, 0, 0, 16)
	if !ok {
		t.Fatal("RangeSpan found no contacts")
	}
	if math.Abs(lo-minVal) > 1e-9 || hi < 9 {
		t.Errorf("RangeSpan gave [%g, %g], want [%g, at least 9]", lo, hi, minVal)
	}
	if _, _, ok := RangeSpan(w, 0, 0, 100, 100); ok {
		t.Error("RangeSpan reported contacts in an empty window")
	}
}

func TestValidateCatchesImpossibleWalks(t *testing.T) {
	for name, mutate := range map[string]func(*Walk){
		"zero speed":      func(w *Walk) { w.Speed = 0 },
		"negative width":  func(w *Walk) { w.Width = -1 },
		"negative stride": func(w *Walk) { w.StrideLength = -1 },
		"impossible gait": func(w *Walk) { w.Stance.Mass = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			w := walker()
			mutate(&w)
			if err := w.Validate(); err == nil {
				t.Errorf("Validate accepted %s", name)
			}
		})
	}
	if err := walker().Validate(); err != nil {
		t.Errorf("Validate rejected a reasonable walk: %v", err)
	}
}

// A walk has to be able to end. An engine that pre-builds a Green's function
// per footfall needs to know there are finitely many; left unbounded it keeps
// discovering footfalls at ever-increasing range and builds a new response for
// each, inside the real-time loop. That is not a theoretical concern — it cost
// a factor of ninety in the per-chunk benchmark before the bound existed.
func TestWalkCanBeBounded(t *testing.T) {
	w := walker()
	w.Until = 5

	all := w.Contacts(0, 100)
	if len(all) == 0 {
		t.Fatal("bounded walk produced no contacts")
	}
	for _, c := range all {
		if c.Start >= w.Until {
			t.Errorf("contact at %g s is past the walk's end at %g s", c.Start, w.Until)
		}
	}
	if got := w.Contacts(w.Until, 100); got != nil {
		t.Errorf("%d contacts after the walk ended", len(got))
	}
	if got := w.Contacts(6, 7); got != nil {
		t.Errorf("%d contacts well after the walk ended", len(got))
	}
	// Unbounded still means unbounded.
	u := walker()
	if got := u.Contacts(500, 510); len(got) == 0 {
		t.Error("an unbounded walk stopped producing contacts")
	}
	// And bounding does not change the contacts that do happen.
	for i, c := range all {
		if u.ContactAt(i) != c {
			t.Errorf("contact %d differs when the walk is bounded", i)
		}
	}
}
