// Command geohier measures the modelling hierarchy and prints the table.
//
//	geohier -model testdata/dispersion/soft_over_stiff.json -ranges 2,5,10,20
//
// What each level costs, and what it costs you. The table is the answer to
// "which model should this part of the system use", which is otherwise decided
// by taste.
//
// The reference is the wavenumber integration at heavily increased sampling.
// It is not an independent check of itself; what makes it trustworthy is V5,
// where a time-domain solver sharing none of its code reproduces it to a
// fraction of a percent. Run geofdtd -compare to see that.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"geosim.dev/geosim/internal/hier"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/units"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "geohier:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		modelPath = flag.String("model", "", "layered model JSON (required)")
		outPath   = flag.String("o", "", "optional CSV of the scores")
		rangeList = flag.String("ranges", "2,5,10,20", "receiver ranges in metres, comma separated")
		band      = flag.Float64("band", 150, "highest frequency scored, Hz")
		bandLow   = flag.Float64("band-low", 5, "lowest frequency scored, Hz")
		q         = flag.Float64("q", 30, "quality factor for layers that do not set one")
		refine    = flag.Int("refine", 8, "reference wavenumber sampling, as a multiple of the default")
	)
	flag.Parse()
	if *modelPath == "" {
		return fmt.Errorf("-model is required")
	}
	g, err := layer.LoadGolden(*modelPath)
	if err != nil {
		return err
	}
	ranges, err := parseRanges(*rangeList)
	if err != nil {
		return err
	}

	slowest, _ := g.Layers.VelocityBounds()
	opt := hier.Options{
		Ranges: ranges, Band: *band, BandLow: *bandLow, Q: *q,
		ReferenceRefinement: *refine,
	}
	opt.Rate, opt.Samples = hier.SuggestGrid(*band, slowest, ranges[len(ranges)-1])

	fmt.Fprintf(os.Stderr, "model %s: %d layer(s), slowest Vs %g m/s\n", g.Name, len(g.Layers), slowest)
	fmt.Fprintf(os.Stderr, "scored over %g-%g Hz on a %g Hz, %d-sample grid\n\n",
		*bandLow, *band, opt.Rate, opt.Samples)

	ref, err := hier.Wavenumber(g.Layers, *q, *refine, "reference", "offline")
	if err != nil {
		return err
	}
	l1, err := hier.Wavenumber(g.Layers, *q, 1, "L1 wavenumber", "offline / bank building")
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "building the bank...\n")
	l1b, _, err := hier.Banked(g.Layers, opt)
	if err != nil {
		return err
	}
	levels := []hier.Level{hier.Analytic(g.Layers, *q), l1, l1b}

	fmt.Fprintf(os.Stderr, "scoring...\n")
	start := time.Now()
	rep, err := hier.Compare(ref, levels, opt)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "done in %v\n\n", time.Since(start).Round(time.Second))

	print(rep, levels)
	if *outPath != "" {
		if err := write(*outPath, rep); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *outPath)
	}
	return nil
}

func parseRanges(s string) ([]units.Metres, error) {
	var out []units.Metres
	for _, part := range strings.Split(s, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("parsing range %q: %w", part, err)
		}
		if v <= 0 {
			return nil, fmt.Errorf("range must be positive, got %g", v)
		}
		out = append(out, units.Metres(v))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no ranges given")
	}
	return out, nil
}

func print(rep *hier.Report, levels []hier.Level) {
	byName := map[string]hier.Level{}
	for _, lv := range levels {
		byName[lv.Name] = lv
	}
	fmt.Printf("%-16s %6s  %10s  %10s  %8s  %9s  %12s\n",
		"level", "range", "median", "worst", "peak", "trace rms", "per response")
	fmt.Println(strings.Repeat("-", 84))
	last := ""
	for _, s := range rep.Scores {
		if s.Level != last && last != "" {
			fmt.Println()
		}
		name := s.Level
		if s.Level == last {
			name = ""
		}
		last = s.Level
		fmt.Printf("%-16s %5.1fm  %9.3f%%  %9.3f%%  %8.4f  %8.3f%%  %12s\n",
			name, s.Range, 100*s.SpectralMedian, 100*s.SpectralMax,
			s.PeakRatio, 100*s.TraceRMS, cost(s.PerResponse))
	}
	fmt.Println()
	for _, lv := range levels {
		line := fmt.Sprintf("  %-16s %s", lv.Name, lv.Where)
		if lv.Setup > 0 {
			line += fmt.Sprintf("; %v to build", lv.Setup.Round(time.Millisecond))
		}
		if lv.Note != "" {
			line += "; " + lv.Note
		}
		fmt.Println(line)
	}
	fmt.Printf("  %-16s %s\n", "L2 grid",
		"offline only; tens of seconds a shot, all ranges at once")
	fmt.Println()
	fmt.Println("  L2 does not appear as a row because it cannot run in the production")
	fmt.Println("  attenuation model — a difference scheme has no fractional power of")
	fmt.Println("  frequency. Its agreement with the reference is V5: 0.1-0.4% on peak")
	fmt.Println("  amplitude extrapolated to zero grid spacing. See geofdtd -compare.")
	fmt.Println()
}

func cost(d time.Duration) string {
	switch {
	case d >= time.Millisecond:
		return fmt.Sprintf("%.2f ms", float64(d)/float64(time.Millisecond))
	case d >= time.Microsecond:
		return fmt.Sprintf("%.2f µs", float64(d)/float64(time.Microsecond))
	}
	return fmt.Sprintf("%d ns", d.Nanoseconds())
}

func write(path string, rep *hier.Report) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write([]string{"level", "range_m", "spectral_median", "spectral_max",
		"worst_at_hz", "peak_ratio", "trace_rms", "per_response_ns"}); err != nil {
		return err
	}
	for _, s := range rep.Scores {
		if err := w.Write([]string{
			s.Level,
			strconv.FormatFloat(float64(s.Range), 'g', 6, 64),
			strconv.FormatFloat(s.SpectralMedian, 'g', 6, 64),
			strconv.FormatFloat(s.SpectralMax, 'g', 6, 64),
			strconv.FormatFloat(s.WorstAt, 'g', 6, 64),
			strconv.FormatFloat(s.PeakRatio, 'g', 6, 64),
			strconv.FormatFloat(s.TraceRMS, 'g', 6, 64),
			strconv.FormatInt(s.PerResponse.Nanoseconds(), 10),
		}); err != nil {
			return err
		}
	}
	return w.Error()
}
