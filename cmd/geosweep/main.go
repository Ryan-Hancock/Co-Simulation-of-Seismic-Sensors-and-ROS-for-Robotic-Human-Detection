// Command geosweep varies one model parameter at a time and reports what it
// does to the signal, writing a CSV for the Python side to plot.
//
// The plan calls the ground reaction force the weakest link in the forward
// model, and says it should be presented as a stated assumption with a
// sensitivity analysis rather than hidden. This is that analysis. It is also
// where O4's domain-randomisation ranges come from: an axis the signal barely
// responds to is not worth randomising over, and one it responds to strongly
// is one the sim-to-real gap will turn on.
//
// One axis at a time, which is the honest limitation to state — it finds the
// gradient at the reference point and says nothing about interactions. A Sobol
// decomposition over the joint space is slice 6's job.
//
//	geosweep -o sweep.csv
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/sensing"
)

// axis is one parameter, the values to try, and how to apply one.
type axis struct {
	name   string
	unit   string
	values []float64
	apply  func(c *config.Config, v float64)
}

func linspace(lo, hi float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = lo + (hi-lo)*float64(i)/float64(n-1)
	}
	return out
}

func axes() []axis {
	return []axis{
		{"body_mass", "kg", linspace(50, 120, 9),
			func(c *config.Config, v float64) { c.Walker.Mass = v }},
		{"walk_speed", "m/s", linspace(0.8, 1.8, 9),
			func(c *config.Config, v float64) { c.Walk.Speed = v }},
		{"transient_rise", "s", linspace(0.006, 0.030, 9),
			func(c *config.Config, v float64) { c.Walker.TransientRise = v }},
		{"transient_peak", "BW", linspace(0.05, 0.70, 9),
			func(c *config.Config, v float64) { c.Walker.TransientPeak = v }},
		{"stance_duration", "s", linspace(0.45, 0.80, 9),
			func(c *config.Config, v float64) { c.Walker.StanceDuration = v }},
		{"ap_peak", "BW", linspace(0.02, 0.40, 9),
			func(c *config.Config, v float64) { c.Walker.APPeak = v }},
		{"shear_velocity", "m/s", linspace(120, 400, 9),
			func(c *config.Config, v float64) {
				c.Soil.Vs = v
				// Hold Poisson's ratio roughly fixed so the axis is velocity
				// and not a confounded velocity-and-compressibility change.
				c.Soil.Vp = 2.5 * v
			}},
		{"soil_q", "", linspace(8, 60, 9),
			func(c *config.Config, v float64) { c.Soil.Qs = v }},
		{"range", "m", linspace(3, 30, 10),
			func(c *config.Config, v float64) { c.Geometry.Range = v }},
		{"coupling_resonance", "Hz", linspace(15, 250, 9),
			func(c *config.Config, v float64) { c.Sensor.CouplingFreq = v }},
	}
}

// metrics are the summary statistics each run is reduced to.
//
// Interpretable statistics rather than sample-wise anything, for the same
// reason WP4 will compare simulated and measured waveforms this way: sample
// agreement is not achievable and not the point.
type metrics struct {
	peakV     float64
	rmsV      float64
	centroidF float64
	snrDB     float64
}

func measure(c config.Config) (metrics, error) {
	res, err := c.Resolve()
	if err != nil {
		return metrics{}, err
	}
	walk := res.WalkPast()
	if err := walk.Validate(); err != nil {
		return metrics{}, err
	}
	fs := res.Sampling.Rate
	n := int(float64(res.WalkDuration()) * fs)
	trace, err := sensing.Reference(res, walk, n)
	if err != nil {
		return metrics{}, err
	}

	var m metrics
	var sumSq float64
	for _, v := range trace {
		m.peakV = math.Max(m.peakV, math.Abs(v))
		sumSq += v * v
	}
	m.rmsV = math.Sqrt(sumSq / float64(len(trace)))

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
		m.centroidF = num / den
	}

	// Against the sensor's own floor, integrated over the band. This is a
	// ceiling on detectability, not a prediction of it: at any real site
	// ambient ground motion is far above the transducer's noise.
	noiseRMS := res.Sensor.NoiseDensity() * math.Sqrt(fs/2)
	if noiseRMS > 0 && m.rmsV > 0 {
		m.snrDB = 20 * math.Log10(m.rmsV/noiseRMS)
	}
	return m, nil
}

func main() {
	out := flag.String("o", "sweep.csv", "output CSV path")
	flag.Parse()

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, "geosweep:", err)
		os.Exit(1)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write([]string{"axis", "unit", "value", "peak_v", "rms_v", "centroid_hz", "snr_db"}); err != nil {
		fmt.Fprintln(os.Stderr, "geosweep:", err)
		os.Exit(1)
	}

	base, err := measure(config.Default())
	if err != nil {
		fmt.Fprintln(os.Stderr, "geosweep: reference run:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "reference: peak %.4g V, rms %.4g V, centroid %.1f Hz, %.0f dB\n",
		base.peakV, base.rmsV, base.centroidF, base.snrDB)

	g := func(v float64) string { return strconv.FormatFloat(v, 'g', 8, 64) }
	for _, a := range axes() {
		var loM, hiM metrics
		for i, v := range a.values {
			c := config.Default()
			a.apply(&c, v)
			m, err := measure(c)
			if err != nil {
				fmt.Fprintf(os.Stderr, "geosweep: %s=%g: %v\n", a.name, v, err)
				continue
			}
			if i == 0 {
				loM = m
			}
			hiM = m
			if err := w.Write([]string{a.name, a.unit, g(v), g(m.peakV), g(m.rmsV), g(m.centroidF), g(m.snrDB)}); err != nil {
				fmt.Fprintln(os.Stderr, "geosweep:", err)
				os.Exit(1)
			}
		}
		fmt.Fprintf(os.Stderr, "%-20s %6.3g -> %-6.3g   rms x%.2f   centroid %.0f -> %.0f Hz\n",
			a.name, a.values[0], a.values[len(a.values)-1],
			hiM.rmsV/loM.rmsV, loM.centroidF, hiM.centroidF)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
}
