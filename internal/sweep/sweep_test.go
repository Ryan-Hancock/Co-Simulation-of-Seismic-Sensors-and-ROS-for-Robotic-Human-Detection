package sweep

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"geosim.dev/geosim/internal/config"
)

// The axes are a claim about the world that O4 will randomise over, so they
// have to be well formed before anything is measured across them. A reversed
// range or a missing setter produces a decomposition that looks entirely
// ordinary.
func TestAxesAreWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, a := range Axes() {
		if a.Name == "" || seen[a.Name] {
			t.Errorf("axis %q: empty or duplicate name", a.Name)
		}
		seen[a.Name] = true
		if a.Lo >= a.Hi {
			t.Errorf("%s: range %g to %g is empty or reversed", a.Name, a.Lo, a.Hi)
		}
		if a.Steps < 2 {
			t.Errorf("%s: %d steps cannot span a range", a.Name, a.Steps)
		}
		if a.Apply == nil {
			t.Errorf("%s: no setter", a.Name)
		}
		if a.Why == "" {
			t.Errorf("%s: no provenance; a range without one is a guess written down", a.Name)
		}
	}
}

// Every axis must actually move the signal at its own endpoints. An axis that
// does not is either wired to nothing or has a range too narrow to matter, and
// both look like a small Sobol index rather than like a mistake.
func TestEachAxisEndpointResolvesAndMoves(t *testing.T) {
	for _, a := range Axes() {
		lo, hi := config.Default(), config.Default()
		a.Apply(&lo, a.Lo)
		a.Apply(&hi, a.Hi)
		mLo, err := Measure(lo)
		if err != nil {
			t.Errorf("%s at %g: %v", a.Name, a.Lo, err)
			continue
		}
		mHi, err := Measure(hi)
		if err != nil {
			t.Errorf("%s at %g: %v", a.Name, a.Hi, err)
			continue
		}
		if mLo.RMSV <= 0 || mHi.RMSV <= 0 {
			t.Errorf("%s: silence at an endpoint", a.Name)
			continue
		}
		moved := math.Abs(mHi.RMSV/mLo.RMSV-1) + math.Abs(mHi.CentroidHz/mLo.CentroidHz-1)
		t.Logf("%-20s rms x%.3f, centroid x%.3f across its range",
			a.Name, mHi.RMSV/mLo.RMSV, mHi.CentroidHz/mLo.CentroidHz)
		if moved < 1e-3 {
			t.Errorf("%s: the signal is unchanged across the whole range; the setter is not connected "+
				"or the range is too narrow to be worth randomising", a.Name)
		}
	}
}

func TestValuesSpanTheRange(t *testing.T) {
	for _, a := range Axes() {
		v := a.Values()
		if len(v) != a.Steps {
			t.Errorf("%s: %d values for %d steps", a.Name, len(v), a.Steps)
		}
		if v[0] != a.Lo || v[len(v)-1] != a.Hi {
			t.Errorf("%s: values run %g to %g, want %g to %g", a.Name, v[0], v[len(v)-1], a.Lo, a.Hi)
		}
	}
}

func TestDesignValidation(t *testing.T) {
	for name, d := range map[string]Design{
		"no columns":   {},
		"unknown axis": {Columns: []string{"not_an_axis"}, Rows: [][]float64{{1}}},
		"duplicate":    {Columns: []string{"body_mass", "body_mass"}, Rows: [][]float64{{1, 2}}},
		"ragged row":   {Columns: []string{"body_mass", "range"}, Rows: [][]float64{{1}}},
	} {
		if err := d.Validate(); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	ok := Design{Columns: []string{"body_mass", "range"}, Rows: [][]float64{{70, 5}}}
	if err := ok.Validate(); err != nil {
		t.Errorf("a valid design was rejected: %v", err)
	}
}

// The batch path runs in parallel and applies axes by name; a run through it
// must be the same run as calling Measure directly. Getting this wrong would
// shuffle results against the rows that produced them, which a variance
// decomposition would absorb as noise rather than report as a fault.
func TestEvaluateMatchesMeasure(t *testing.T) {
	d := Design{
		Columns: []string{"body_mass", "shear_velocity", "range"},
		Rows: [][]float64{
			{60, 150, 4},
			{95, 300, 12},
			{110, 220, 8},
		},
	}
	got, errs := Evaluate(d)
	for i, row := range d.Rows {
		if errs[i] != nil {
			t.Fatalf("row %d: %v", i, errs[i])
		}
		c := config.Default()
		for j, name := range d.Columns {
			a, _ := ByName(name)
			a.Apply(&c, row[j])
		}
		want, err := Measure(c)
		if err != nil {
			t.Fatal(err)
		}
		if got[i] != want {
			t.Errorf("row %d: batch gave %+v, direct gave %+v", i, got[i], want)
		}
	}
}

// A Sobol design deliberately visits the corners of the box, and some corners
// are not media. One rejected row must not take the batch with it, and must not
// be quietly replaced by a nearby value either — a fabricated point inside a
// variance decomposition is indistinguishable from a real one.
func TestARejectedRowDoesNotStopTheBatch(t *testing.T) {
	d := Design{
		Columns: []string{"body_mass"},
		Rows:    [][]float64{{70}, {-5}, {80}},
	}
	m, errs := Evaluate(d)
	if errs[1] == nil {
		t.Error("a negative body mass was accepted")
	}
	if m[1] != (Metrics{}) {
		t.Errorf("the rejected row carries metrics %+v; it should carry none", m[1])
	}
	for _, i := range []int{0, 2} {
		if errs[i] != nil {
			t.Errorf("row %d failed alongside the bad row: %v", i, errs[i])
		}
		if m[i].RMSV <= 0 {
			t.Errorf("row %d produced no signal", i)
		}
	}
}

func TestDesignRoundTripsThroughCSV(t *testing.T) {
	d := Design{
		Columns: []string{"body_mass", "range"},
		Rows:    [][]float64{{70, 5}, {90, 12.5}},
	}
	m := make([]Metrics, len(d.Rows))
	errs := make([]error, len(d.Rows))
	m[0] = Metrics{PeakV: 1e-3, RMSV: 2e-4, CentroidHz: 33, SNRdB: 40}
	errs[1] = errTest{}

	var buf bytes.Buffer
	if err := WriteResults(&buf, d, m, errs); err != nil {
		t.Fatal(err)
	}
	text := buf.String()
	// The design travels with its results, so a results file cannot be read
	// against the wrong design.
	if !strings.Contains(text, "body_mass,range,peak_v,rms_v,centroid_hz,snr_db,error") {
		t.Errorf("header does not carry both the design and the metrics:\n%s", text)
	}
	if !strings.Contains(text, "deliberate") {
		t.Errorf("the row error was not written:\n%s", text)
	}

	back, err := ReadDesign(strings.NewReader(text))
	if err == nil {
		t.Errorf("a results file parsed as a design; its extra columns should be rejected, got %v", back.Columns)
	}

	plain := "body_mass,range\n70,5\n90,12.5\n"
	got, err := ReadDesign(strings.NewReader(plain))
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Rows) != 2 || got.Rows[1][1] != 12.5 {
		t.Errorf("round trip lost values: %+v", got)
	}
}

type errTest struct{}

func (errTest) Error() string { return "deliberate" }

func TestReadDesignRejectsRubbish(t *testing.T) {
	for name, text := range map[string]string{
		"empty":        "",
		"header only":  "body_mass\n",
		"not a number": "body_mass\nheavy\n",
	} {
		if _, err := ReadDesign(strings.NewReader(text)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
