// Package sweep is the parameter space of the forward model, and what it does
// to a signal.
//
// It exists as a package rather than as part of a command because slice 6 needs
// the same axes and the same measurement from two directions: the
// one-at-a-time sweep that slice 2 built, and a Sobol decomposition over the
// joint space. Two copies of a parameter range is how a sensitivity analysis
// ends up characterising a model nobody is running.
//
// The Sobol design is generated on the Python side and evaluated here, which
// keeps the language seam where §6 puts it: Python writes a file, Go reads it,
// and there is no per-sample boundary between them. A Sobol analysis is tens of
// thousands of evaluations, so a per-call boundary is exactly the thing that
// would make it unaffordable.
package sweep

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/sensing"
)

// Axis is one parameter: what it is called, the range it is credible over, and
// how to set it.
//
// Lo and Hi are the ranges O4 will randomise over, so they are a claim about
// the world rather than a plotting convenience. They are stated here once and
// used by both the sweep and the decomposition.
type Axis struct {
	Name string  `json:"name"`
	Unit string  `json:"unit"`
	Lo   float64 `json:"lo"`
	Hi   float64 `json:"hi"`
	// Steps is how many values the one-at-a-time sweep tries.
	Steps int `json:"steps"`
	// Why records where the range comes from, because a range without a
	// provenance is a guess that has been written down.
	Why string `json:"why"`

	Apply func(*config.Config, float64) `json:"-"`
}

// Values are the one-at-a-time sweep's samples.
func (a Axis) Values() []float64 {
	out := make([]float64, a.Steps)
	for i := range out {
		out[i] = a.Lo + (a.Hi-a.Lo)*float64(i)/float64(a.Steps-1)
	}
	return out
}

// Axes is the parameter space.
func Axes() []Axis {
	return []Axis{
		{Name: "body_mass", Unit: "kg", Lo: 50, Hi: 120, Steps: 9,
			Why:   "adult range; the GRF scales with it directly",
			Apply: func(c *config.Config, v float64) { c.Walker.Mass = v }},
		{Name: "walk_speed", Unit: "m/s", Lo: 0.8, Hi: 1.8, Steps: 9,
			Why:   "slow amble to brisk walk, short of running",
			Apply: func(c *config.Config, v float64) { c.Walk.Speed = v }},
		{Name: "transient_rise", Unit: "s", Lo: 0.006, Hi: 0.030, Steps: 9,
			Why:   "heel-strike rise times from force-plate literature",
			Apply: func(c *config.Config, v float64) { c.Walker.TransientRise = v }},
		{Name: "transient_peak", Unit: "BW", Lo: 0.05, Hi: 0.70, Steps: 9,
			Why:   "heel-strike spike amplitude, which varies hugely with footwear and surface",
			Apply: func(c *config.Config, v float64) { c.Walker.TransientPeak = v }},
		{Name: "stance_duration", Unit: "s", Lo: 0.45, Hi: 0.80, Steps: 9,
			Why:   "stance phase at walking cadences",
			Apply: func(c *config.Config, v float64) { c.Walker.StanceDuration = v }},
		{Name: "ap_peak", Unit: "BW", Lo: 0.02, Hi: 0.40, Steps: 9,
			Why:   "fore-aft shear as a fraction of body weight",
			Apply: func(c *config.Config, v float64) { c.Walker.APPeak = v }},
		{Name: "shear_velocity", Unit: "m/s", Lo: 120, Hi: 400, Steps: 9,
			Why: "dry sand through firm soil; the dominant medium parameter",
			Apply: func(c *config.Config, v float64) {
				c.Soil.Vs = v
				// Hold Poisson's ratio roughly fixed so the axis is velocity
				// and not a confounded velocity-and-compressibility change.
				c.Soil.Vp = 2.5 * v
			}},
		{Name: "soil_q", Unit: "", Lo: 8, Hi: 60, Steps: 9,
			Why:   "wet loose soil through firm ground",
			Apply: func(c *config.Config, v float64) { c.Soil.Qs = v }},
		{Name: "range", Unit: "m", Lo: 3, Hi: 30, Steps: 10,
			Why:   "the detection ranges O1 cares about",
			Apply: func(c *config.Config, v float64) { c.Geometry.Range = v }},
		{Name: "coupling_resonance", Unit: "Hz", Lo: 15, Hi: 250, Steps: 9,
			Why:   "a spike pushed into soft ground through one bedded on rock",
			Apply: func(c *config.Config, v float64) { c.Sensor.CouplingFreq = v }},
	}
}

// ByName looks up an axis.
func ByName(name string) (Axis, bool) {
	for _, a := range Axes() {
		if a.Name == name {
			return a, true
		}
	}
	return Axis{}, false
}

// Metrics are the summary statistics each run is reduced to.
//
// Interpretable statistics rather than sample-wise anything, for the same
// reason WP4 will compare simulated and measured waveforms this way: sample
// agreement is not achievable and not the point.
type Metrics struct {
	PeakV      float64
	RMSV       float64
	CentroidHz float64
	SNRdB      float64
}

// Names are the metric column names, in the order Values returns them.
func (Metrics) Names() []string { return []string{"peak_v", "rms_v", "centroid_hz", "snr_db"} }

// Values are the metrics as a slice, in the order Names gives.
func (m Metrics) Values() []float64 { return []float64{m.PeakV, m.RMSV, m.CentroidHz, m.SNRdB} }

// Measure runs one configuration and reduces it to metrics.
func Measure(c config.Config) (Metrics, error) {
	res, err := c.Resolve()
	if err != nil {
		return Metrics{}, err
	}
	walk := res.WalkPast()
	if err := walk.Validate(); err != nil {
		return Metrics{}, err
	}
	fs := res.Sampling.Rate
	n := int(float64(res.WalkDuration()) * fs)
	trace, err := sensing.Reference(res, walk, n)
	if err != nil {
		return Metrics{}, err
	}

	var m Metrics
	var sumSq float64
	for _, v := range trace {
		m.PeakV = math.Max(m.PeakV, math.Abs(v))
		sumSq += v * v
	}
	m.RMSV = math.Sqrt(sumSq / float64(len(trace)))

	// Spectral centroid over the band a footstep detector would use. It is the
	// statistic that moves when the source's high-frequency content changes,
	// which is exactly what the transient parameters control.
	padded := make([]float64, dsp.NextPow2(len(trace)))
	copy(padded, trace)
	coeff := dsp.RFFT(padded)
	var num, den float64
	for k, f := range dsp.FreqBins(len(padded), fs) {
		if f < 2 || f > 400 {
			continue
		}
		a := math.Hypot(real(coeff[k]), imag(coeff[k]))
		num += f * a
		den += a
	}
	if den > 0 {
		m.CentroidHz = num / den
	}

	// Against the sensor's own floor, integrated over the band. This is a
	// ceiling on detectability, not a prediction of it: at any real site
	// ambient ground motion is far above the transducer's noise.
	noiseRMS := res.Sensor.NoiseDensity() * math.Sqrt(fs/2)
	if noiseRMS > 0 && m.RMSV > 0 {
		m.SNRdB = 20 * math.Log10(m.RMSV/noiseRMS)
	}
	return m, nil
}

// Design is a batch of parameter vectors to evaluate: one column per named
// axis, one row per run. Axes not named keep their default value.
type Design struct {
	Columns []string
	Rows    [][]float64
}

// Validate reports whether every column names a real axis and every row is the
// right width.
func (d Design) Validate() error {
	if len(d.Columns) == 0 {
		return fmt.Errorf("sweep: a design needs at least one column")
	}
	seen := map[string]bool{}
	for _, c := range d.Columns {
		if _, ok := ByName(c); !ok {
			return fmt.Errorf("sweep: no axis called %q", c)
		}
		if seen[c] {
			return fmt.Errorf("sweep: column %q appears twice", c)
		}
		seen[c] = true
	}
	for i, r := range d.Rows {
		if len(r) != len(d.Columns) {
			return fmt.Errorf("sweep: row %d has %d values for %d columns", i, len(r), len(d.Columns))
		}
	}
	return nil
}

// Evaluate runs a whole design.
//
// A row that cannot be resolved — a combination the config rejects as
// physically impossible — yields an error at that index rather than stopping
// the batch. A Sobol design deliberately visits corners of the box, and some
// corners are not media; losing the whole run to one of them would be the wrong
// trade, and silently substituting a nearby value would put a fabricated point
// into a variance decomposition.
func Evaluate(d Design) ([]Metrics, []error) {
	out := make([]Metrics, len(d.Rows))
	errs := make([]error, len(d.Rows))
	axes := make([]Axis, len(d.Columns))
	for i, c := range d.Columns {
		axes[i], _ = ByName(c)
	}

	work := make(chan int, runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				c := config.Default()
				for j, a := range axes {
					a.Apply(&c, d.Rows[i][j])
				}
				out[i], errs[i] = Measure(c)
			}
		}()
	}
	for i := range d.Rows {
		work <- i
	}
	close(work)
	wg.Wait()
	return out, errs
}
