package geophone

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/units"
)

// isolated is an SM-24 with the coupling removed, so tests of the transducer
// see only the transducer.
func isolated() Geophone {
	g := SM24()
	g.Coupling = PerfectCoupling()
	return g
}

// V7: the transducer response against its closed form at the three
// frequencies where it has one.
//
//	f >> f0:  |H| -> G, phase -> 0
//	f == f0:  |H| == G/(2*zeta), phase == +90 degrees
//	f << f0:  |H| -> G*(f/f0)^2, i.e. 12 dB/octave
func TestTransducerAnalyticLimits(t *testing.T) {
	g := isolated()
	G := float64(g.Sensitivity)
	zeta := g.Damping()

	t.Run("flat above resonance", func(t *testing.T) {
		h := g.TransducerResponse(500)
		if rel := math.Abs(cmplx.Abs(h)-G) / G; rel > 1e-3 {
			t.Errorf("|H(500 Hz)| = %g, want %g (rel err %g)", cmplx.Abs(h), G, rel)
		}
		if ph := cmplx.Phase(h) * 180 / math.Pi; math.Abs(ph) > 1 {
			t.Errorf("phase at 500 Hz = %g degrees, want ~0", ph)
		}
	})

	t.Run("at resonance", func(t *testing.T) {
		h := g.TransducerResponse(g.NaturalFreq)
		want := G / (2 * zeta)
		if rel := math.Abs(cmplx.Abs(h)-want) / want; rel > 1e-12 {
			t.Errorf("|H(f0)| = %g, want %g", cmplx.Abs(h), want)
		}
		if ph := cmplx.Phase(h) * 180 / math.Pi; math.Abs(ph-90) > 1e-9 {
			t.Errorf("phase at f0 = %g degrees, want +90", ph)
		}
	})

	t.Run("twelve dB per octave below resonance", func(t *testing.T) {
		// Well below f0 the response is G*(f/f0)^2, so halving the frequency
		// must drop it by a factor of four.
		lo, hi := g.TransducerResponse(0.1), g.TransducerResponse(0.2)
		ratio := cmplx.Abs(hi) / cmplx.Abs(lo)
		if math.Abs(ratio-4) > 0.05 {
			t.Errorf("octave ratio = %g, want 4 (12 dB/octave)", ratio)
		}
		want := G * math.Pow(0.1/float64(g.NaturalFreq), 2)
		if rel := math.Abs(cmplx.Abs(lo)-want) / want; rel > 5e-3 {
			t.Errorf("|H(0.1 Hz)| = %g, want %g (rel err %g)", cmplx.Abs(lo), want, rel)
		}
	})

	t.Run("no response at DC", func(t *testing.T) {
		if h := g.TransducerResponse(0); cmplx.Abs(h) != 0 {
			t.Errorf("|H(0)| = %g, want 0: a velocity transducer cannot see a static load", cmplx.Abs(h))
		}
	})
}

// A shunt adds damping and divides signal, and the two must move together:
// ShuntForDamping is the inverse of Damping, so composing them is identity.
func TestShuntDampingRoundTrip(t *testing.T) {
	g := isolated()
	for _, want := range []float64{0.5, 0.6, 0.7, 0.9} {
		rs, err := g.ShuntForDamping(want)
		if err != nil {
			t.Fatalf("damping %g: %v", want, err)
		}
		loaded := g
		loaded.ShuntResistance = rs
		if got := loaded.Damping(); math.Abs(got-want) > 1e-12 {
			t.Errorf("shunt %g ohm gives damping %g, want %g", rs, got, want)
		}
	}
}

// The canonical field configuration — an SM-24 shunted to 0.7 of critical —
// should land on a load of a few kilohms. A model that returned ohms or
// megohms here would be dimensionally adrift somewhere in Damping.
func TestShuntForSeventyPercentIsPlausible(t *testing.T) {
	rs, err := SM24().ShuntForDamping(0.7)
	if err != nil {
		t.Fatal(err)
	}
	if rs < 1000 || rs > 10000 {
		t.Errorf("shunt for 0.7 damping = %g ohm, expected a few kilohm", rs)
	}
}

func TestShuntCannotReduceDamping(t *testing.T) {
	g := SM24()
	if _, err := g.ShuntForDamping(g.OpenCircuitDamping); err == nil {
		t.Error("expected an error: a shunt cannot damp less than open circuit")
	}
	if _, err := g.ShuntForDamping(0.1); err == nil {
		t.Error("expected an error for damping below the open-circuit value")
	}
}

// Coupling is unity at DC and rolls off above its resonance. A well-planted
// spike must be near-transparent across the footstep band, or the model would
// be attributing the sensor's mounting to the soil.
func TestCouplingTransparentInBand(t *testing.T) {
	g := SM24()
	for _, f := range []units.Hertz{1, 10, 50, 100} {
		mag := cmplx.Abs(g.CouplingResponse(f))
		if mag < 0.95 || mag > 1.35 {
			t.Errorf("well-planted coupling at %g Hz = %g, want near unity in band", f, mag)
		}
	}
	if mag := cmplx.Abs(g.CouplingResponse(2000)); mag > 0.05 {
		t.Errorf("coupling at 2 kHz = %g, want the case to have decoupled", mag)
	}
	if mag := cmplx.Abs(PerfectCoupling().response(10)); mag != 1 {
		t.Errorf("perfect coupling at 10 Hz = %g, want exactly 1", mag)
	}
}

// A soft-ground resonance inside the band is the case O3 has to live with:
// it must visibly distort, or the parameter is not doing anything.
func TestPoorCouplingDistortsInBand(t *testing.T) {
	g := SM24()
	g.Coupling = Coupling{ResonanceFreq: 30, Damping: 0.2}
	if mag := cmplx.Abs(g.CouplingResponse(30)); mag < 2 {
		t.Errorf("poor coupling at its 30 Hz resonance = %g, want a clear peak", mag)
	}
	if mag := cmplx.Abs(g.CouplingResponse(100)); mag > 0.2 {
		t.Errorf("poor coupling at 100 Hz = %g, want strong attenuation above resonance", mag)
	}
}

// Apply must agree with Response: filtering a tone through the time-domain
// path should give the amplitude the frequency response predicts.
func TestApplyMatchesResponse(t *testing.T) {
	g := isolated()
	const fs = 1000.0
	for _, f := range []float64{5, 20, 80} {
		n := int(fs * 4)
		v := make([]float64, n)
		for i := range v {
			v[i] = math.Sin(2 * math.Pi * f * float64(i) / fs)
		}
		out, err := g.Apply(v, fs)
		if err != nil {
			t.Fatal(err)
		}
		// Measure in the steady state, away from the filter's turn-on.
		var peak float64
		for _, s := range out[n/2:] {
			peak = math.Max(peak, math.Abs(float64(s)))
		}
		want := cmplx.Abs(g.Response(units.Hertz(f)))
		if rel := math.Abs(peak-want) / want; rel > 5e-3 {
			t.Errorf("%g Hz: Apply gives peak %g V, Response predicts %g V (rel err %g)", f, peak, want, rel)
		}
	}
}

func TestApplyRejectsBadInput(t *testing.T) {
	g := SM24()
	if _, err := g.Apply([]float64{1, 2}, 0); err == nil {
		t.Error("expected an error for a zero sample rate")
	}
	bad := SM24()
	bad.CoilResistance = 0
	if _, err := bad.Apply([]float64{1, 2}, 1000); err == nil {
		t.Error("expected an error for an invalid geophone")
	}
	out, err := g.Apply(nil, 1000)
	if err != nil || out != nil {
		t.Errorf("empty input: got %v, %v; want nil, nil", out, err)
	}
}

// V8: Johnson noise. The density must be sqrt(4kTR) for the coil alone, and
// the parallel combination once a shunt is fitted.
func TestNoiseDensityJohnson(t *testing.T) {
	g := isolated()

	open := math.Sqrt(4 * units.Boltzmann * units.RoomTemperatureK * float64(g.CoilResistance))
	if got := g.NoiseDensity(); math.Abs(got-open)/open > 1e-12 {
		t.Errorf("open-circuit density = %g V/rtHz, want %g", got, open)
	}
	// The published figure for a 375 ohm coil at room temperature.
	if got := g.NoiseDensity(); got < 2.4e-9 || got > 2.6e-9 {
		t.Errorf("open-circuit density = %g V/rtHz, want ~2.5 nV/rtHz", got)
	}

	loaded := g
	loaded.ShuntResistance = 3300
	rc, rs := float64(g.CoilResistance), 3300.0
	par := math.Sqrt(4 * units.Boltzmann * units.RoomTemperatureK * rc * rs / (rc + rs))
	if got := loaded.NoiseDensity(); math.Abs(got-par)/par > 1e-12 {
		t.Errorf("shunted density = %g V/rtHz, want the parallel value %g", got, par)
	}
	if loaded.NoiseDensity() >= g.NoiseDensity() {
		t.Error("a shunt should lower the output noise density, not raise it")
	}
}

// Referred to ground velocity the SM-24's own floor should sit in the tens of
// picometres per second per root hertz — orders of magnitude below any real
// site's ambient motion. This number is the basis of the O5 argument that
// detection range is set by the site and not the transducer, so it is worth
// pinning rather than recomputing by hand later.
func TestNoiseFloorInVelocityIsWellBelowAmbient(t *testing.T) {
	got := isolated().NoiseDensityInVelocity()
	if got < 5e-11 || got > 2e-10 {
		t.Errorf("equivalent input noise = %g (m/s)/rtHz, want ~8.7e-11", got)
	}
}

// V8: the generated time series must actually have the density claimed.
// Checked through Parseval rather than a periodogram — the total energy is a
// far lower-variance estimator than any single bin.
func TestNoiseSeriesHasStatedDensity(t *testing.T) {
	g := isolated()
	const (
		fs = 2000.0
		n  = 1 << 16
	)
	noise := g.Noise(n, fs, rand.New(rand.NewPCG(7, 8)))

	raw := make([]float64, n)
	for i, v := range noise {
		raw[i] = float64(v)
	}
	// One-sided PSD, flat: total energy / (bandwidth * n) gives density^2.
	density := math.Sqrt(dsp.Energy(raw) / float64(n) / (fs / 2))
	want := g.NoiseDensity()
	if rel := math.Abs(density-want) / want; rel > 0.02 {
		t.Errorf("measured density %g V/rtHz, want %g (rel err %g)", density, want, rel)
	}
}

func TestNoiseIsReproducible(t *testing.T) {
	g := SM24()
	a := g.Noise(64, 1000, rand.New(rand.NewPCG(1, 2)))
	b := g.Noise(64, 1000, rand.New(rand.NewPCG(1, 2)))
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed gave different noise at sample %d: %g vs %g", i, a[i], b[i])
		}
	}
}

func TestValidateCatchesNonsense(t *testing.T) {
	for name, mutate := range map[string]func(*Geophone){
		"zero natural frequency": func(g *Geophone) { g.NaturalFreq = 0 },
		"zero sensitivity":       func(g *Geophone) { g.Sensitivity = 0 },
		"zero coil resistance":   func(g *Geophone) { g.CoilResistance = 0 },
		"zero moving mass":       func(g *Geophone) { g.MovingMass = 0 },
		"zero damping":           func(g *Geophone) { g.OpenCircuitDamping = 0 },
		"negative coupling":      func(g *Geophone) { g.Coupling.ResonanceFreq = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			g := SM24()
			mutate(&g)
			if err := g.Validate(); err == nil {
				t.Errorf("Validate accepted %s", name)
			}
		})
	}
	if err := SM24().Validate(); err != nil {
		t.Errorf("Validate rejected the reference SM-24: %v", err)
	}
}
