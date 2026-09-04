package gait

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/units"
)

// The one identity a gait cannot escape: speed is step length times step
// frequency. Everything else here is a relation read off the literature and
// could be wrong by a bit; this one is a definition, and getting it wrong means
// the walker slides.
//
// It caught a factor of ten. The walk ratio is stated per step per *minute* and
// the frequency is wanted per second, and dropping the sixty inside the square
// root gave an 11 m stride — obvious once printed, invisible in a test that only
// checked the stride grew with speed.
func TestSpeedIsStepLengthTimesCadence(t *testing.T) {
	for v := 0.6; v <= 2.1; v += 0.1 {
		g, err := At(units.SpeedMPS(v))
		if err != nil {
			t.Fatal(err)
		}
		got := float64(g.StepLength) * g.StepFrequency
		if math.Abs(got-v) > 1e-12*v {
			t.Errorf("at %.1f m/s the gait implies %.6f m/s", v, got)
		}
		if float64(g.StrideLength) != 2*float64(g.StepLength) {
			t.Errorf("at %.1f m/s a stride is not two steps", v)
		}
		if math.Abs(float64(g.StanceDuration)-g.DutyFactor*float64(g.CycleTime)) > 1e-12 {
			t.Errorf("at %.1f m/s the stance is not the duty factor of the cycle", v)
		}
	}
}

// The numbers a gait lab would report, at the speeds it would report them.
// These are the relations' whole content, so they are checked against the
// published ranges rather than against each other.
func TestGaitMatchesTheLiterature(t *testing.T) {
	for _, c := range []struct {
		v                  float64
		cadLo, cadHi       float64 // steps per minute
		strideLo, strideHi float64 // metres
		stanceLo, stanceHi float64 // seconds
	}{
		{0.8, 80, 92, 1.00, 1.20, 0.80, 0.95},
		{1.3, 104, 116, 1.30, 1.50, 0.62, 0.72},
		{1.8, 122, 134, 1.60, 1.75, 0.50, 0.60},
	} {
		g, err := At(units.SpeedMPS(c.v))
		if err != nil {
			t.Fatal(err)
		}
		cad := 60 * g.StepFrequency
		t.Logf("%.1f m/s: %.1f steps/min, stride %.3f m, stance %.3f s, duty %.3f, double support %.3f s",
			c.v, cad, g.StrideLength, g.StanceDuration, g.DutyFactor, g.DoubleSupport)
		if cad < c.cadLo || cad > c.cadHi {
			t.Errorf("%.1f m/s: cadence %.1f steps/min, want %g to %g", c.v, cad, c.cadLo, c.cadHi)
		}
		if s := float64(g.StrideLength); s < c.strideLo || s > c.strideHi {
			t.Errorf("%.1f m/s: stride %.3f m, want %g to %g", c.v, s, c.strideLo, c.strideHi)
		}
		if s := float64(g.StanceDuration); s < c.stanceLo || s > c.stanceHi {
			t.Errorf("%.1f m/s: stance %.3f s, want %g to %g", c.v, s, c.stanceLo, c.stanceHi)
		}
	}
}

func TestEverythingMovesTheRightWayWithSpeed(t *testing.T) {
	var prev Gait
	var prevStance grf.Stance
	for v := 0.6; v <= 2.1; v += 0.1 {
		g, err := At(units.SpeedMPS(v))
		if err != nil {
			t.Fatal(err)
		}
		s, err := g.Stance(grf.Stance{Mass: 75})
		if err != nil {
			t.Fatal(err)
		}
		if prev.StepFrequency > 0 {
			switch {
			case g.StepFrequency <= prev.StepFrequency:
				t.Errorf("at %.1f m/s cadence did not rise", v)
			case g.StepLength <= prev.StepLength:
				t.Errorf("at %.1f m/s step length did not rise", v)
			case g.DutyFactor >= prev.DutyFactor:
				t.Errorf("at %.1f m/s the duty factor did not fall", v)
			case g.StanceDuration >= prev.StanceDuration:
				t.Errorf("at %.1f m/s the stance did not shorten", v)
			case g.DoubleSupport >= prev.DoubleSupport:
				t.Errorf("at %.1f m/s double support did not shorten", v)
			case s.FirstPeak <= prevStance.FirstPeak:
				t.Errorf("at %.1f m/s the first peak did not rise", v)
			case s.MidstanceValley >= prevStance.MidstanceValley:
				t.Errorf("at %.1f m/s the midstance valley did not deepen", v)
			case s.APPeak <= prevStance.APPeak:
				t.Errorf("at %.1f m/s the fore-aft shear did not rise", v)
			}
		}
		prev, prevStance = g, s
	}
}

// The momentum balance is not fitted here, it is solved, so it has to hold
// exactly at every speed rather than approximately near one.
//
// This is what the whole arrangement is for. The demanded impulse goes as one
// over the duty factor, which falls by a tenth across the walking range, so a
// profile balanced at one speed is eight percent out at the other end — and
// that error is entirely in the low-frequency content of the radiated field,
// where no amount of correct propagation modelling recovers it.
func TestMomentumBalancesAtEverySpeed(t *testing.T) {
	for v := 0.6; v <= 2.1; v += 0.05 {
		g, err := At(units.SpeedMPS(v))
		if err != nil {
			t.Fatal(err)
		}
		s, err := g.Stance(grf.Stance{Mass: 75})
		if err != nil {
			t.Fatal(err)
		}
		if r := s.ImpulseRatio(); math.Abs(r-1) > 1e-5 {
			t.Errorf("at %.2f m/s the impulse ratio is %.6f, not 1", v, r)
		}
	}
}

// The solved valley is a prediction, not a parameter, so it is worth checking
// that it lands where force plates put it — and worth knowing if it does not,
// because then either the peak relations or the balance is wrong.
func TestSolvedValleyIsPhysiological(t *testing.T) {
	for _, v := range []float64{0.8, 1.3, 1.8} {
		g, _ := At(units.SpeedMPS(v))
		s, err := g.Stance(grf.Stance{Mass: 75})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%.1f m/s: peaks %.3f and %.3f, solved valley %.3f BW",
			v, s.FirstPeak, s.SecondPeak, s.MidstanceValley)
		if s.MidstanceValley < 0.55 || s.MidstanceValley > 0.90 {
			t.Errorf("%.1f m/s: the balance wants a valley of %.3f BW, outside the 0.55 to 0.90 "+
				"force plates report; the peak relations and the balance disagree", v, s.MidstanceValley)
		}
		if s.MidstanceValley >= s.FirstPeak || s.MidstanceValley >= s.SecondPeak {
			t.Errorf("%.1f m/s: the valley is not below the peaks", v)
		}
	}
}

// The transient is left uncoupled on purpose, and a caller's choice of it must
// survive being handed to the gait.
func TestFootwearSurvivesTheGait(t *testing.T) {
	g, _ := At(1.3)
	base := grf.Stance{Mass: 80, TransientPeak: 0.55, TransientRise: 0.008, HumpWidth: 0.12}
	s, err := g.Stance(base)
	if err != nil {
		t.Fatal(err)
	}
	if s.TransientPeak != base.TransientPeak || s.TransientRise != base.TransientRise ||
		s.HumpWidth != base.HumpWidth || s.Mass != base.Mass {
		t.Errorf("the gait overwrote something it does not own: %+v", s)
	}
	// And the balance still holds, because the valley was solved with the
	// transient's own impulse already in the sum.
	if r := s.ImpulseRatio(); math.Abs(r-1) > 1e-5 {
		t.Errorf("impulse ratio %.6f with a heavy heel strike", r)
	}
}

func TestRefusesGaitsItCannotDescribe(t *testing.T) {
	for _, v := range []units.SpeedMPS{0, 0.3, 2.5, -1} {
		if _, err := At(v); err == nil {
			t.Errorf("%g m/s was accepted", v)
		}
	}
}
