package grf

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/units"
)

// The strongest available check on a fitted profile, and the one that
// constrains its free parameters: over a gait cycle in steady walking the
// centre of mass returns to the same height with the same velocity, so the
// ground's vertical impulse from both feet must equal the weight impulse
// exactly. That is Newton's second law integrated over a period.
//
// A profile can look entirely convincing and still fail this, and if it does
// it is handing the ground the wrong momentum — which corrupts the
// low-frequency content of the radiated field in a way that no amount of
// correct propagation modelling downstream can recover.
func TestImpulseBalancesBodyWeightOverGaitCycle(t *testing.T) {
	for _, mass := range []units.Kilograms{50, 75, 95, 120} {
		s := Walker(mass)
		if got := s.ImpulseRatio(); math.Abs(got-1) > 1e-3 {
			t.Errorf("mass %g kg: impulse ratio %.5f, want 1", mass, got)
		}
	}
}

// The balance must survive the profile being varied, since O4 will randomise
// these. It degrades away from the defaults — the taper was tuned at the
// reference walker — so the tolerance here is the honest modelling error, not
// a claim of exactness.
func TestImpulseBalanceAcrossPlausibleGaits(t *testing.T) {
	for _, s := range []Stance{
		{Mass: 75, Duration: 0.50, FirstPeak: 1.30, SecondPeak: 1.25, MidstanceValley: 0.65},
		{Mass: 60, Duration: 0.70, FirstPeak: 1.08, SecondPeak: 1.06, MidstanceValley: 0.85},
		{Mass: 90, Duration: 0.62, FirstPeak: 1.20, SecondPeak: 1.15, MidstanceValley: 0.72},
	} {
		got := s.ImpulseRatio()
		if math.Abs(got-1) > 0.06 {
			t.Errorf("%+v: impulse ratio %.4f, want within 6%% of 1", s, got)
		}
	}
}

// V11: the profile hits the peaks and valley the literature reports, at the
// stance phases it reports them at.
func TestProfileShapeMatchesForcePlateLiterature(t *testing.T) {
	s := Walker(75)

	if got := s.ProfileAt(firstPeakPhase); math.Abs(got-s.FirstPeak) > 0.01 {
		t.Errorf("first peak = %.4f BW at tau=0.25, want %.2f", got, s.FirstPeak)
	}
	if got := s.ProfileAt(secondPeakPhase); math.Abs(got-s.SecondPeak) > 0.01 {
		t.Errorf("second peak = %.4f BW at tau=0.75, want %.2f", got, s.SecondPeak)
	}
	// The valley sits slightly above the requested level, lifted by the tails
	// of both humps. That is the correct behaviour for a sum of overlapping
	// components, not an error to tune away.
	if got := s.ProfileAt(0.5); got < s.MidstanceValley || got > s.MidstanceValley+0.02 {
		t.Errorf("midstance = %.4f BW, want just above %.2f", got, s.MidstanceValley)
	}

	// Peaks are maxima and midstance is a minimum — the M shape, checked
	// rather than assumed.
	if s.ProfileAt(0.5) >= s.ProfileAt(0.25) || s.ProfileAt(0.5) >= s.ProfileAt(0.75) {
		t.Error("midstance is not a trough between the two peaks")
	}
	// The transient's tail still rings faintly through the loading phase, so
	// the realised peak sits within a percent of the nominal hump rather than
	// exactly on it.
	nominal := s.FirstPeak * float64(s.BodyWeight())
	if peak := float64(s.PeakForce()); math.Abs(peak-nominal)/nominal > 0.02 {
		t.Errorf("PeakForce = %g N, want within 2%% of the nominal first peak %g N", peak, nominal)
	}
}

// The force must reach zero continuously at heel strike and toe-off. A profile
// that starts at a non-zero value is a step, and a step's spectrum falls off
// only as 1/f — it would radiate across the whole detection band and swamp the
// signal it was supposed to carry.
//
// Continuity, not slowness. Once the heel-strike transient is present the force
// rises fast on purpose: that rise is the physical impact, and it is most of
// what a geophone at range sees. What matters is that it starts from zero.
func TestForceIsContinuousAtContactBoundaries(t *testing.T) {
	s := Walker(75)

	for _, tau := range []float64{-0.1, -1e-9, 0, 1, 1 + 1e-9, 1.5} {
		if got := s.ProfileAt(tau); got != 0 {
			t.Errorf("tau=%g: force %.6g BW, want exactly 0 outside stance", tau, got)
		}
	}
	// Approaching heel contact from inside, the force must fall to zero rather
	// than jump. A step would show as a value that stops shrinking as tau does.
	prev := math.Inf(1)
	for _, tau := range []float64{1e-4, 1e-5, 1e-6, 1e-7, 1e-8} {
		got := s.ProfileAt(tau)
		if got <= 0 || got >= prev {
			t.Errorf("tau=%g: force %.3e BW did not shrink below the value at the previous decade (%.3e)", tau, got, prev)
		}
		prev = got
	}
	if got := s.ProfileAt(1e-8); got > 1e-6 {
		t.Errorf("force at tau=1e-8 is %.3e BW, want negligible: the profile is stepping, not rising", got)
	}
	// And the same at toe-off.
	for _, tau := range []float64{1 - 1e-6, 1 - 1e-8} {
		if got := s.ProfileAt(tau); got > 1e-4 {
			t.Errorf("tau=%g: force %.3e BW, want a continuous fall to zero at toe-off", tau, got)
		}
	}
}

// The heel-strike transient is the part that matters most for detection at
// range: the smooth double hump is quasi-static and carries almost nothing
// above a few hertz. These are the properties that make it that.
func TestHeelStrikeTransient(t *testing.T) {
	s := Walker(75)

	t.Run("reaches its stated peak at the stated rise time", func(t *testing.T) {
		// The parameter is the height the transient actually reaches, so it
		// can be read off a published force trace rather than back-computed.
		var peak, peakTau float64
		for i := 1; i < 20000; i++ {
			tau := float64(i) / 20000
			if v := s.transientAt(tau); v > peak {
				peak, peakTau = v, tau
			}
		}
		if math.Abs(peak-defaultTransientPeak) > 1e-3 {
			t.Errorf("transient peak %.4f BW, want %.4f", peak, defaultTransientPeak)
		}
		wantTau := float64(s.transientRise()) / float64(s.Duration)
		if math.Abs(peakTau-wantTau)/wantTau > 0.15 {
			t.Errorf("transient peaks at tau=%.4f (%.1f ms), want near the rise time %.1f ms",
				peakTau, peakTau*float64(s.Duration)*1000, s.transientRise()*1000)
		}
	})

	t.Run("dominates the first tens of milliseconds", func(t *testing.T) {
		// At one rise time in, the force must already be a large fraction of a
		// body weight. This is the assertion that catches the transient being
		// swallowed by the smooth curve's taper — which it was, until the
		// taper was applied to the smooth part alone. The force trace still
		// looked like a footstep with the transient suppressed fourfold; only
		// its radiated spectrum would have shown the loss.
		tau := float64(s.transientRise()) / float64(s.Duration)
		if got := s.ProfileAt(tau); got < 0.30 {
			t.Errorf("force at one rise time is %.4f BW, want above 0.30: the transient is being suppressed", got)
		}
	})

	t.Run("rings down within about fifty milliseconds", func(t *testing.T) {
		late := 0.080 / float64(s.Duration) // 80 ms in
		if got := math.Abs(s.transientAt(late)); got > 0.05 {
			t.Errorf("transient is still %.4f BW at 80 ms, want it rung down", got)
		}
	})

	t.Run("can be switched off", func(t *testing.T) {
		off := Walker(75)
		off.TransientPeak = -1
		for _, tau := range []float64{0.01, 0.02, 0.05} {
			if got := off.transientAt(tau); got != 0 {
				t.Errorf("tau=%g: transient %g with TransientPeak negative, want 0", tau, got)
			}
		}
	})
}

// Fore-aft shear brakes through the first half of stance and propels through
// the second, and over the whole stance it must integrate to zero: a walker
// holding a steady speed changes no fore-aft momentum from step to step.
//
// It is included because a horizontal surface force excites Rayleigh waves too,
// with a different radiation pattern and a quarter-cycle phase shift. Dropping
// it is a simplification usually made without being stated.
func TestAnteriorPosteriorShear(t *testing.T) {
	s := Walker(75)

	if got := s.AnteriorPosteriorAt(0.25); got > -0.19 || got < -0.21 {
		t.Errorf("braking peak %.4f BW at tau=0.25, want about -0.20", got)
	}
	if got := s.AnteriorPosteriorAt(0.75); got < 0.19 || got > 0.21 {
		t.Errorf("propulsion peak %.4f BW at tau=0.75, want about +0.20", got)
	}
	if got := s.AnteriorPosteriorAt(0.5); math.Abs(got) > 1e-9 {
		t.Errorf("shear at midstance is %.3e BW, want a zero crossing", got)
	}

	const n = 200000
	var sum float64
	for i := range n {
		sum += s.AnteriorPosteriorAt((float64(i) + 0.5) / n)
	}
	if mean := sum / n; math.Abs(mean) > 1e-4 {
		t.Errorf("mean fore-aft shear %.3e BW, want zero: braking must cancel propulsion", mean)
	}

	for _, tau := range []float64{-0.1, 0, 1, 1.5} {
		if got := s.AnteriorPosteriorAt(tau); got != 0 {
			t.Errorf("tau=%g: shear %g outside stance, want 0", tau, got)
		}
	}
	off := Walker(75)
	off.APPeak = -1
	if got := off.AnteriorPosteriorAt(0.25); got != 0 {
		t.Errorf("shear %g with APPeak negative, want 0", got)
	}
}

// Sample must place the stance where the lead says it does, and be silent
// either side of it.
func TestSamplePlacesStanceCorrectly(t *testing.T) {
	const fs = 1000.0
	s := Walker(75)
	lead, tail := units.Seconds(0.2), units.Seconds(0.3)

	got, err := s.Sample(fs, lead, tail)
	if err != nil {
		t.Fatal(err)
	}
	wantN := int(math.Round((float64(lead) + float64(s.Duration) + float64(tail)) * fs))
	if len(got) != wantN {
		t.Fatalf("length %d, want %d", len(got), wantN)
	}
	for i := range int(float64(lead) * fs) {
		if got[i] != 0 {
			t.Fatalf("sample %d is %g N before heel strike, want silence", i, got[i])
		}
	}
	// One sample clear of toe-off, since the boundary sample can land either
	// side of it under floating-point rounding.
	for i := int(math.Ceil((float64(lead)+float64(s.Duration))*fs)) + 1; i < len(got); i++ {
		if got[i] != 0 {
			t.Fatalf("sample %d is %g N after toe-off, want silence", i, got[i])
		}
	}
	// The peak lands a quarter of the way into stance.
	var peakAt int
	var peak float64
	for i, v := range got {
		if float64(v) > peak {
			peak, peakAt = float64(v), i
		}
	}
	wantAt := (float64(lead) + firstPeakPhase*float64(s.Duration)) * fs
	if math.Abs(float64(peakAt)-wantAt) > 5 {
		t.Errorf("peak at sample %d, want near %.0f", peakAt, wantAt)
	}
	if rel := math.Abs(peak-float64(s.PeakForce())) / peak; rel > 2e-3 {
		t.Errorf("sampled peak force %g N, want PeakForce's %g N", peak, s.PeakForce())
	}
}

// Peak force scales with mass and nothing else does anything surprising.
func TestForceScalesWithBodyWeight(t *testing.T) {
	light, heavy := Walker(50), Walker(100)
	if ratio := float64(heavy.PeakForce()) / float64(light.PeakForce()); math.Abs(ratio-2) > 1e-9 {
		t.Errorf("doubling mass scaled peak force by %g, want 2", ratio)
	}
	if got := float64(Walker(75).BodyWeight()); math.Abs(got-75*units.GravityMPS2) > 1e-9 {
		t.Errorf("body weight %g N, want %g", got, 75*units.GravityMPS2)
	}
}

func TestValidateCatchesImplausibleGaits(t *testing.T) {
	for name, mutate := range map[string]func(*Stance){
		"zero mass":       func(s *Stance) { s.Mass = 0 },
		"zero duration":   func(s *Stance) { s.Duration = 0 },
		"zero first peak": func(s *Stance) { s.FirstPeak = 0 },
		"zero valley":     func(s *Stance) { s.MidstanceValley = 0 },
		"valley above peak": func(s *Stance) {
			s.MidstanceValley = 1.5
		},
		"absurd hump width": func(s *Stance) { s.HumpWidth = 0.9 },
		"absurd taper":      func(s *Stance) { s.TaperFraction = 0.8 },
	} {
		t.Run(name, func(t *testing.T) {
			s := Walker(75)
			mutate(&s)
			if err := s.Validate(); err == nil {
				t.Errorf("Validate accepted %s", name)
			}
			if _, err := s.Sample(1000, 0, 0); err == nil {
				t.Errorf("Sample accepted %s", name)
			}
		})
	}
	if err := Walker(75).Validate(); err != nil {
		t.Errorf("Validate rejected the reference walker: %v", err)
	}
	if _, err := Walker(75).Sample(0, 0, 0); err == nil {
		t.Error("Sample accepted a zero sample rate")
	}
	if _, err := Walker(75).Sample(1000, -1, 0); err == nil {
		t.Error("Sample accepted a negative lead")
	}
}
