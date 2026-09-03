package sensing

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/bank"
	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/fk"
	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// testBank builds a bank small enough for a test but through the real path.
func testBank(t *testing.T, rMin, rMax float64, maxFreq float64) *bank.Bank {
	t.Helper()
	stack := layer.Uniform(soil.Loam())
	m := fk.Medium{Stack: stack, DefaultQ: 30}

	slowest, _ := stack.VelocityBounds()
	limit := bank.RangeNyquist(0.87*float64(slowest), maxFreq) / 2
	count := int(math.Ceil((rMax-rMin)/limit)) + 1

	h := bank.Header{
		FormatVersion: bank.FormatVersion,
		Provenance:    bank.Provenance{Solver: "test"},
		Medium:        stack,
		SampleRateHz:  2000,
		Samples:       1024,
		Ranges:        bank.RangeGrid{MinM: rMin, MaxM: rMax, Count: count},
		Component:     "vertical surface displacement per unit vertical surface point force",
		Units:         "m/N",
	}
	if err := h.CheckRangeSampling(maxFreq); err != nil {
		t.Fatal(err)
	}
	b, err := bank.New(h)
	if err != nil {
		t.Fatal(err)
	}
	ranges := make([]units.Metres, count)
	for i := range ranges {
		ranges[i] = units.Metres(h.Ranges.At(i))
	}
	top := int(maxFreq / h.FrequencyAt(1))
	for k := 1; k <= top && k < h.Bins(); k++ {
		vals, err := m.VerticalDisplacementMulti(ranges, h.FrequencyAt(k), fk.Integration{})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range vals {
			b.Set(i, k, v)
		}
	}
	return b
}

// The point of slice 5: the layered wavenumber physics reaching the synthesis
// path, where until now only slice 0's far-field closed form could go.
//
// The engine takes either model through the same interface, so a run can be
// repeated with better physics and nothing else changed — which is what makes
// the comparison below a measurement rather than two different experiments.
func TestBankDrivesSynthesis(t *testing.T) {
	b := testBank(t, 1, 12, 80)
	prop, err := FromBank(b)
	if err != nil {
		t.Fatal(err)
	}

	c := config.Default()
	c.Geometry.Range = 4
	c.Geometry.ApproachLength = 4
	res, err := c.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	walk := res.WalkPast()

	e, err := NewEngineWith(res, walk, 20, prop)
	if err != nil {
		t.Fatal(err)
	}
	if e.Propagation() == "" {
		t.Error("the engine does not report its propagation model")
	}

	n := int(float64(res.WalkDuration()) * res.Sampling.Rate)
	buf := make([]float32, 20)
	var trace []float64
	for len(trace) < n {
		if err := e.Next(buf); err != nil {
			t.Fatal(err)
		}
		for _, v := range buf {
			trace = append(trace, float64(v))
		}
	}

	var peak float64
	for _, v := range trace[:n] {
		peak = math.Max(peak, math.Abs(v))
	}
	if peak == 0 {
		t.Fatal("bank-driven synthesis produced silence")
	}
	// Microvolts to millivolts for a walker a few metres away: the same order
	// the analytic path gives, which is the check that the units survived the
	// displacement-to-velocity conversion and the interpolation.
	if peak < 1e-6 || peak > 1e-1 {
		t.Errorf("peak output %g V is not a believable geophone signal", peak)
	}
}

// The interface has to be honest about what a bank does not carry. A bank holds
// one component, and mixing a near-field-correct vertical response with a
// far-field analytic horizontal one would be incoherent — the whole reason for
// the bank is that the far-field form is wrong at these ranges. Slice 2's
// sensitivity sweep is what licenses omitting the shear: varying it over a
// factor of twenty moved a walk-past by under a percent.
func TestBankOmitsTheShearDeliberately(t *testing.T) {
	b := testBank(t, 1, 6, 60)
	prop, err := FromBank(b)
	if err != nil {
		t.Fatal(err)
	}
	got, err := prop.RadialForceResponse(3, 20)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Errorf("bank propagation returned a shear response of %v; it holds only the vertical component", got)
	}
	// The analytic model does carry it, so the interface is not simply unable
	// to express one.
	an := Analytic(green.HalfSpaceGF{Soil: soil.Loam()})
	if v, err := an.RadialForceResponse(3, 20); err != nil || v == 0 {
		t.Errorf("the analytic model should carry a shear response, got %v (%v)", v, err)
	}
}

// A bank defines the frequency grid its responses live on, and the engine has
// to invert on that same grid rather than one of its own choosing. Asking off
// the grid is an error rather than a silent interpolation, because interpolating
// in frequency across a response that oscillates with period c/r would be
// bridging a curve the grid barely resolves.
func TestBankRejectsOffGridFrequencies(t *testing.T) {
	b := testBank(t, 1, 6, 60)
	prop, err := FromBank(b)
	if err != nil {
		t.Fatal(err)
	}
	spacing := b.FrequencyAt(1)
	if _, err := prop.VerticalVelocityResponse(3, units.Hertz(spacing*4)); err != nil {
		t.Errorf("an on-grid frequency was rejected: %v", err)
	}
	if _, err := prop.VerticalVelocityResponse(3, units.Hertz(spacing*4.5)); err == nil {
		t.Error("expected an off-grid frequency to be rejected")
	}
	if got, err := prop.VerticalVelocityResponse(3, 0); err != nil || got != 0 {
		t.Errorf("DC should be zero without error, got %v (%v)", got, err)
	}
	if prop.TransformSize() != b.Samples {
		t.Errorf("transform size %d, want the bank's %d", prop.TransformSize(), b.Samples)
	}
	if _, err := FromBank(nil); err == nil {
		t.Error("expected an error for a nil bank")
	}
}
