// Command geobank builds and inspects Green's function banks.
//
//	geobank build -medium testdata/dispersion/soft_over_stiff.json -o soil.bank
//	geobank inspect soil.bank
//
// Building runs the wavenumber integration once per frequency and shares those
// samples across every range, which is what makes a bank affordable: the
// expensive part of the calculation does not depend on range at all.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"geosim.dev/geosim/internal/bank"
	"geosim.dev/geosim/internal/fk"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: geobank build|inspect ...")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "build":
		err = build(os.Args[2:])
	case "inspect":
		err = inspect(os.Args[2:])
	default:
		err = fmt.Errorf("unknown command %q (have build, inspect)", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "geobank:", err)
		os.Exit(1)
	}
}

func build(args []string) error {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	var (
		mediumPath = fs.String("medium", "", "layered medium JSON; empty uses the loam half-space")
		out        = fs.String("o", "green.bank", "output path")
		rate       = fs.Float64("rate", 2000, "sample rate the bank will be inverted at, Hz")
		samples    = fs.Int("samples", 4096, "transform length; sets the frequency grid and the response duration")
		maxFreq    = fs.Float64("max-freq", 300, "highest frequency modelled; above this the bank is zero")
		rMin       = fs.Float64("r-min", 1, "closest range, m")
		rMax       = fs.Float64("r-max", 40, "furthest range, m")
		spacing    = fs.Float64("spacing", 0, "range spacing in m; zero picks the coarsest that can still be interpolated")
		q          = fs.Float64("q", 30, "quality factor for layers that do not set one")
	)
	fs.Parse(args)

	stack, err := loadMedium(*mediumPath)
	if err != nil {
		return err
	}
	m := fk.Medium{Stack: stack, DefaultQ: *q}

	// The range grid has to satisfy the phase-unwrapping condition, and there
	// is no reason to make it finer than that: a denser grid costs build time
	// and disk for accuracy the interpolation does not need.
	slowest, _ := stack.VelocityBounds()
	limit := bank.RangeNyquist(0.87*float64(slowest), *maxFreq) / 2
	step := *spacing
	if step <= 0 {
		step = limit
	}
	count := int(math.Ceil((*rMax-*rMin)/step)) + 1
	if count < 2 {
		count = 2
	}

	h := bank.Header{
		FormatVersion: bank.FormatVersion,
		Provenance: bank.Provenance{
			Solver:  "geosim fk (layered wavenumber integration)",
			Created: time.Now().UTC().Format(time.RFC3339),
			Notes:   fmt.Sprintf("band-limited to %g Hz; Q=%g where unset", *maxFreq, *q),
		},
		Medium:       stack,
		SampleRateHz: *rate,
		Samples:      *samples,
		Ranges:       bank.RangeGrid{MinM: *rMin, MaxM: *rMax, Count: count},
		Component:    "vertical surface displacement per unit vertical surface point force",
		Units:        "m/N",
	}
	if err := h.Validate(); err != nil {
		return err
	}
	if err := h.CheckRangeSampling(*maxFreq); err != nil {
		return err
	}

	b, err := bank.New(h)
	if err != nil {
		return err
	}
	ranges := make([]units.Metres, count)
	for i := range ranges {
		ranges[i] = units.Metres(h.Ranges.At(i))
	}

	// Bins above the modelled band are left zero: attenuation has removed
	// anything there, and computing them would dominate the build.
	topBin := int(*maxFreq / (*rate / float64(*samples)))
	if topBin >= h.Bins() {
		topBin = h.Bins() - 1
	}

	fmt.Fprintf(os.Stderr, "medium: %d layer(s), slowest Vs %g m/s\n", len(stack), slowest)
	fmt.Fprintf(os.Stderr, "ranges: %d from %g to %g m, spacing %.4f m (limit %.4f m)\n",
		count, *rMin, *rMax, h.Ranges.Spacing(), limit)
	fmt.Fprintf(os.Stderr, "bins:   %d of %d modelled (up to %g Hz)\n", topBin, h.Bins(), *maxFreq)

	start := time.Now()
	var done atomic.Int64
	var mu sync.Mutex
	var firstErr error

	workers := runtime.GOMAXPROCS(0)
	binCh := make(chan int, workers)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := range binCh {
				freq := h.FrequencyAt(k)
				vals, err := m.VerticalDisplacementMulti(ranges, freq, fk.Integration{})
				mu.Lock()
				if err != nil {
					if firstErr == nil {
						firstErr = fmt.Errorf("bin %d (%.3f Hz): %w", k, freq, err)
					}
					mu.Unlock()
					continue
				}
				for i, v := range vals {
					b.Set(i, k, v)
				}
				mu.Unlock()
				if n := done.Add(1); n%50 == 0 {
					fmt.Fprintf(os.Stderr, "\r  %d/%d bins", n, topBin)
				}
			}
		}()
	}
	// Bin 0 is DC, where a static point load gives a finite but physically
	// irrelevant offset and the eigenbasis is degenerate. Left zero.
	for k := 1; k <= topBin; k++ {
		binCh <- k
	}
	close(binCh)
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	fmt.Fprintf(os.Stderr, "\r  %d/%d bins in %s\n", done.Load(), topBin, time.Since(start).Round(time.Millisecond))

	if err := bank.Write(*out, b); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%.2f MB)\n", *out, float64(b.SizeBytes())/(1<<20))
	return nil
}

func inspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ExitOnError)
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: geobank inspect <bank>")
	}
	b, err := bank.Open(fs.Arg(0))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(b.Header); err != nil {
		return err
	}
	fmt.Printf("payload: %.2f MB, %d ranges x %d bins\n",
		float64(b.SizeBytes())/(1<<20), b.Ranges.Count, b.Bins())
	fmt.Printf("range spacing: %.4f m\n", b.Ranges.Spacing())
	fmt.Printf("frequency spacing: %.4f Hz, Nyquist %.1f Hz\n",
		b.FrequencyAt(1), b.FrequencyAt(b.Bins()-1))
	return nil
}

// loadMedium reads a layered medium, or falls back to the loam half-space.
func loadMedium(path string) (layer.Stack, error) {
	if path == "" {
		return layer.Uniform(soil.Loam()), nil
	}
	g, err := layer.LoadGolden(path)
	if err != nil {
		return nil, err
	}
	return g.Layers, nil
}
