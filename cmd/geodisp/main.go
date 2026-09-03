// Command geodisp computes Rayleigh dispersion curves for a layered medium and
// writes them as CSV.
//
//	geodisp -model testdata/dispersion/three_layer_site.json -o curves.csv
//
// The model is read from the same JSON the golden files use, so a curve can be
// computed for exactly the medium a golden file describes and the two compared
// directly.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"

	"geosim.dev/geosim/internal/disp"
	"geosim.dev/geosim/internal/layer"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "geodisp:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		modelPath = flag.String("model", "", "layered model JSON (required)")
		outPath   = flag.String("o", "dispersion.csv", "output CSV path")
		modes     = flag.Int("modes", 3, "how many modes to attempt")
		fLo       = flag.Float64("f-lo", 2, "lowest frequency, Hz")
		fHi       = flag.Float64("f-hi", 120, "highest frequency, Hz")
		points    = flag.Int("points", 120, "frequency samples, logarithmically spaced")
	)
	flag.Parse()
	if *modelPath == "" {
		return fmt.Errorf("-model is required")
	}

	g, err := layer.LoadGolden(*modelPath)
	if err != nil {
		return err
	}
	freqs := make([]float64, *points)
	for i := range freqs {
		freqs[i] = *fLo * math.Pow(*fHi / *fLo, float64(i)/float64(*points-1))
	}

	f, err := os.Create(*outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write([]string{"mode", "frequency_hz", "phase_velocity_mps", "group_velocity_mps"}); err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "%s: %d layer(s), %s\n", g.Name, len(g.Layers), g.Why)
	lo, hi := disp.Bounds(g.Layers)
	fmt.Fprintf(os.Stderr, "search bounds %.1f to %.1f m/s\n", lo, hi)

	num := func(v float64) string { return strconv.FormatFloat(v, 'g', 8, 64) }
	for mode := range *modes {
		curve, err := disp.PhaseCurve(g.Layers, freqs, mode, disp.Search{})
		if err != nil {
			fmt.Fprintf(os.Stderr, "mode %d: %v\n", mode, err)
			continue
		}
		for i, fr := range curve.Frequency {
			// Group velocity can fail at the very edges of a mode's existence,
			// where the neighbouring frequencies it needs fall outside.
			gv, err := disp.GroupVelocity(g.Layers, fr, mode, disp.Search{})
			gs := ""
			if err == nil {
				gs = num(float64(gv))
			}
			if err := w.Write([]string{strconv.Itoa(mode), num(fr), num(float64(curve.PhaseVelocity[i])), gs}); err != nil {
				return err
			}
		}
		fmt.Fprintf(os.Stderr, "mode %d: %d points, %.1f to %.1f m/s\n", mode,
			len(curve.Frequency), float64(curve.PhaseVelocity[len(curve.PhaseVelocity)-1]), float64(curve.PhaseVelocity[0]))
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *outPath)
	return nil
}
