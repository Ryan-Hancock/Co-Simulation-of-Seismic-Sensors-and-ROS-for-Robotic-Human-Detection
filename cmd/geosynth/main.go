// Command geosynth synthesises the geophone trace a single footstep produces
// at a given range over a given soil.
//
// This is slice 0 of the WP1 plan end to end: a ground reaction force becomes
// a Rayleigh wave becomes a voltage. Everything it does is deliberately the
// simplest defensible version — one foot, one homogeneous half-space, far
// field only — so that the later slices have a working baseline to replace one
// stage at a time rather than a blank page.
//
//	geosynth -o trace.csv
//	geosynth -config run.json -o trace.csv
//	geosynth -print-config > run.json
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/sensing"
	"geosim.dev/geosim/internal/source"
	"geosim.dev/geosim/internal/units"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "geosynth:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		configPath  = flag.String("config", "", "JSON run description; the built-in default is used if empty")
		outPath     = flag.String("o", "trace.csv", "output CSV path")
		rangeM      = flag.Float64("range", 0, "override the source-receiver range, in metres")
		printConfig = flag.Bool("print-config", false, "write the default configuration to stdout and exit")
		quiet       = flag.Bool("quiet", false, "suppress the summary on stderr")
	)
	flag.Parse()

	if *printConfig {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(config.Default())
	}

	cfg := config.Default()
	if *configPath != "" {
		data, err := os.ReadFile(*configPath)
		if err != nil {
			return err
		}
		if cfg, err = config.Parse(data); err != nil {
			return err
		}
	}
	if *rangeM > 0 {
		cfg.Geometry.Range = *rangeM
	}

	res, err := cfg.Resolve()
	if err != nil {
		return err
	}
	hash, err := res.Hash()
	if err != nil {
		return err
	}

	// A walk past the sensor: each footfall lands somewhere different, so each
	// has its own range and its own arrival. The rise and fall that produces is
	// the signature, not an artefact of it.
	walk := res.WalkPast()
	if err := walk.Validate(); err != nil {
		return err
	}
	fs := res.Sampling.Rate
	n := int((float64(res.WalkDuration()) + res.Sampling.Tail) * fs)

	raw, err := sensing.Reference(res, walk, n)
	if err != nil {
		return err
	}
	volts := make([]units.Volts, len(raw))
	for i, v := range raw {
		volts[i] = units.Volts(v)
	}

	// The total vertical force on the ground, summed over whatever feet are
	// down, recorded alongside so the trace can be read against its own source.
	force := make([]units.Newtons, len(raw))
	nearest := make([]float64, len(raw))
	for i := range nearest {
		nearest[i] = math.Inf(1)
	}
	for _, c := range walk.Contacts(0, units.Seconds(float64(n)/fs)) {
		r := math.Hypot(c.X, c.Y)
		for i := range force {
			at := units.Seconds(float64(i) / fs)
			f := c.ForceAt(at)
			force[i] += units.Newtons(f[2])
			if f[2] != 0 && r < nearest[i] {
				nearest[i] = r
			}
		}
	}
	for i, v := range nearest {
		if math.IsInf(v, 1) {
			nearest[i] = 0
		}
	}

	// Sensor noise is added after the transfer function, because that is where
	// it originates: it is the coil's own Johnson noise, not ground motion.
	var noise []units.Volts
	if res.Noise.Enabled {
		noise = res.Sensor.Noise(len(volts), res.Sampling.Rate, rand.New(rand.NewPCG(res.Noise.Seed, 0x5eed)))
		for i := range volts {
			volts[i] += noise[i]
		}
	}

	if err := writeCSV(*outPath, res, force, nearest, volts, noise); err != nil {
		return err
	}
	if err := writeSidecar(*outPath+".json", res, hash); err != nil {
		return err
	}

	if !*quiet {
		printSummary(res, hash, walk, force, volts, *outPath)
	}
	return nil
}

func writeCSV(path string, res config.Resolved, force []units.Newtons, rangeM []float64, volts, noise []units.Volts) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{"t_s", "force_n", "range_m", "volts"}
	if noise != nil {
		header = append(header, "noise_v")
	}
	if err := w.Write(header); err != nil {
		return err
	}

	dt := 1 / res.Sampling.Rate
	g := func(v float64) string { return strconv.FormatFloat(v, 'g', 10, 64) }
	// The three series have different lengths — the force record is the
	// shortest, since propagation adds travel time and coda. Writing to the
	// longest and zero-filling keeps the CSV rectangular.
	for i := range volts {
		var fN float64
		if i < len(force) {
			fN = float64(force[i])
		}
		row := []string{g(float64(i) * dt), g(fN), g(rangeM[i]), g(float64(volts[i]))}
		if noise != nil {
			row = append(row, g(float64(noise[i])))
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return w.Error()
}

// writeSidecar records the resolved configuration and its hash next to the
// trace, so a run can be reproduced from its own output rather than from
// whatever the shell history happens to remember.
func writeSidecar(path string, res config.Resolved, hash string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(struct {
		ConfigHash string          `json:"config_hash"`
		Resolved   config.Resolved `json:"resolved"`
	}{hash, res})
}

func printSummary(res config.Resolved, hash string, walk source.Walk, force []units.Newtons, volts []units.Volts, out string) {
	cr, _ := res.Soil.RayleighVelocity()
	gf := green.HalfSpaceGF{Soil: res.Soil}
	travel, _ := gf.TravelTime(units.Metres(res.Geometry.Range))

	peak := func(f func(int) float64, n int) float64 {
		var m float64
		for i := range n {
			m = math.Max(m, math.Abs(f(i)))
		}
		return m
	}
	peakF := peak(func(i int) float64 { return float64(force[i]) }, len(force))
	peakE := peak(func(i int) float64 { return float64(volts[i]) }, len(volts))

	fmt.Fprintf(os.Stderr, "config   %s\n", hash[:16])
	fmt.Fprintf(os.Stderr, "soil     %s  cR=%.1f m/s\n", res.Soil, cr)
	fmt.Fprintf(os.Stderr, "walker   %.0f kg, stance %.2f s, impulse ratio %.4f\n",
		res.Walker.Mass, res.Walker.Duration, res.Walker.ImpulseRatio())
	fmt.Fprintf(os.Stderr, "sensor   f0=%.1f Hz  damping %.3f  shunt %.0f ohm  floor %.2g (m/s)/rtHz\n",
		res.Sensor.NaturalFreq, res.Sensor.Damping(), res.Sensor.ShuntResistance, res.Sensor.NoiseDensityInVelocity())
	fmt.Fprintf(os.Stderr, "walk     %.2f m/s, step %.3f s, stride %.2f m, %d footfalls over %.1f s\n",
		res.Walk.Speed, walk.StepPeriod(), walk.StrideOrDefault(),
		len(walk.Contacts(0, res.WalkDuration())), res.WalkDuration())
	fmt.Fprintf(os.Stderr, "geometry closest approach %.1f m, travel %.4f s\n", res.Geometry.Range, travel)
	fmt.Fprintf(os.Stderr, "peaks    force %.1f N -> %.4g V\n", peakF, peakE)
	if res.Noise.Enabled {
		fmt.Fprintf(os.Stderr, "         signal-to-noise at peak: %.0f dB above the sensor's own floor\n",
			20*math.Log10(peakE/(res.Sensor.NoiseDensity()*math.Sqrt(res.Sampling.Rate/2))))
	}
	fmt.Fprintf(os.Stderr, "wrote    %s (%d samples, %.2f s) and %s.json\n",
		out, len(volts), float64(len(volts))/res.Sampling.Rate, out)
}
