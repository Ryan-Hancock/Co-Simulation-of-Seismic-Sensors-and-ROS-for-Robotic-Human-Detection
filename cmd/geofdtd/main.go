// Command geofdtd runs the time-domain solver and, with -compare, the
// frequency-wavenumber path beside it.
//
//	geofdtd -model testdata/dispersion/two_layer_site.json -ranges 3,6,10 -compare -o v5.csv
//
// The comparison is validation V5, and this is how it is reproduced outside the
// test suite: one command, a CSV of both traces at each range, and a printed
// table of how far apart they are.
//
// Two settings decide whether the answer means anything. The grid spacing has
// to resolve the shortest wavelength in the band — -ppw sets how finely, and
// the scheme is first-order accurate at the free surface, so the error falls
// only linearly with it. And -ref-samples has to be large enough that the
// reference itself is converged: the wavenumber integrand has a near-pole at
// the Rayleigh wavenumber, the trapezoidal rule is first-order accurate there,
// and the default sampling leaves about two percent at the top of the band.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"strconv"
	"strings"
	"time"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/fdtd"
	"geosim.dev/geosim/internal/fk"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/units"
	"geosim.dev/geosim/internal/visco"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "geofdtd:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		modelPath  = flag.String("model", "", "layered model JSON (required)")
		outPath    = flag.String("o", "fdtd.csv", "output CSV path")
		rangeList  = flag.String("ranges", "4,8,16", "receiver ranges in metres, comma separated")
		f0         = flag.Float64("f0", 25, "Ricker wavelet peak frequency, Hz")
		band       = flag.Float64("band", 0, "highest frequency to resolve, Hz (default 3.5*f0)")
		ppw        = flag.Float64("ppw", 20, "grid cells across the shortest wavelength")
		spacing    = flag.Float64("spacing", 0, "grid spacing in metres, overriding -ppw")
		seconds    = flag.Float64("seconds", 0.3, "how long to run")
		peak       = flag.Float64("force", 1000, "wavelet peak force, N")
		q          = flag.Float64("q", 25, "quality factor at the reference frequency; 0 for elastic")
		refFreq    = flag.Float64("ref-freq", 30, "frequency at which nominal velocities are phase velocities, Hz")
		depth      = flag.Float64("depth", 0, "domain depth in metres (default 1.2x the furthest range)")
		margin     = flag.Float64("margin", 1.5, "domain radius as a multiple of the furthest range")
		compare    = flag.Bool("compare", false, "also synthesise through the frequency-wavenumber path")
		refSamples = flag.Int("ref-samples", 160000, "wavenumber samples for the reference")
	)
	flag.Parse()
	if *modelPath == "" {
		return fmt.Errorf("-model is required")
	}
	if *band <= 0 {
		*band = 3.5 * *f0
	}

	g, err := layer.LoadGolden(*modelPath)
	if err != nil {
		return err
	}
	ranges, err := parseRanges(*rangeList)
	if err != nil {
		return err
	}
	furthest := ranges[len(ranges)-1]

	var relax visco.SLS
	if *q > 0 {
		if relax, err = visco.Fit(*q, *refFreq); err != nil {
			return err
		}
	} else if *compare {
		return fmt.Errorf("-compare needs attenuation: with no loss the Rayleigh pole " +
			"sits on the wavenumber integration path and the reference does not exist")
	}

	m := fdtd.Model{
		Stack: g.Layers, Relax: relax, RefFreq: units.Hertz(*refFreq),
		MaxRange:     units.Metres(*margin * float64(furthest)),
		Depth:        units.Metres(math.Max(*depth, 1.2*float64(furthest))),
		DominantFreq: units.Hertz(*f0),
	}
	if *spacing > 0 {
		m.Spacing = units.Metres(*spacing)
	} else if m.Spacing, err = m.SpacingFor(units.Hertz(*band), *ppw); err != nil {
		return err
	}

	probe, err := fdtd.New(m)
	if err != nil {
		return err
	}
	dt := probe.Dt()
	steps := int(*seconds / dt)
	nr, nz := probe.Cells()
	fmt.Fprintf(os.Stderr, "model %s: %d layers, spacing %.4f m, grid %dx%d, dt %.3e s, %d steps\n",
		g.Name, len(g.Layers), m.Spacing, nr, nz, dt, steps)

	force, err := fdtd.Ricker(units.Newtons(*peak), units.Hertz(*f0), dt, steps)
	if err != nil {
		return err
	}
	start := time.Now()
	res, err := fdtd.Run(fdtd.Shot{
		Model: m, Force: force, Ranges: ranges, Steps: steps,
		Progress: func(step, total int) {
			fmt.Fprintf(os.Stderr, "\r  %3d%%", 100*step/total)
		},
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\r  grid run %.1f s\n", time.Since(start).Seconds())

	snapped := make([]units.Metres, len(res.Traces))
	for i, tr := range res.Traces {
		snapped[i] = tr.Range
	}
	var ref [][]float64
	if *compare {
		start = time.Now()
		if ref, err = synthesise(g.Layers, relax, units.Hertz(*refFreq), snapped,
			units.Newtons(*peak), units.Hertz(*f0), *band, dt, res.T0, steps, *refSamples); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "  reference %.1f s\n", time.Since(start).Seconds())
	}

	if err := write(*outPath, res, ref); err != nil {
		return err
	}
	if ref != nil {
		summarise(res, ref)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *outPath)
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

// synthesise drives the frequency-wavenumber path with the same wavelet, on the
// same instants the grid recorded.
func synthesise(st layer.Stack, sls visco.SLS, refFreq units.Hertz, ranges []units.Metres,
	peak units.Newtons, f0 units.Hertz, band, dt, t0 float64, n, samples int) ([][]float64, error) {
	m := fk.Medium{Stack: st, Relax: &sls, RefFreq: refFreq}
	nfft := dsp.NextPow2(n)
	fs := 1 / dt
	df := fs / float64(nfft)
	coeff := make([][]complex128, len(ranges))
	for i := range coeff {
		coeff[i] = make([]complex128, nfft/2+1)
	}
	for b := 1; b <= nfft/2; b++ {
		f := float64(b) * df
		if f > band {
			break
		}
		u, err := m.VerticalDisplacementMulti(ranges, f, fk.Integration{Samples: samples})
		if err != nil {
			return nil, err
		}
		w := fdtd.RickerSpectrum(peak, f0, f)
		jw := complex(0, 2*math.Pi*f)
		shift := cmplx.Exp(complex(0, -2*math.Pi*f*t0))
		for i := range ranges {
			coeff[i][b] = u[i] * w * jw * shift * complex(fs, 0)
		}
	}
	out := make([][]float64, len(ranges))
	for i := range ranges {
		out[i] = dsp.IRFFT(coeff[i], nfft)[:n]
	}
	return out, nil
}

func write(path string, res *fdtd.Result, ref [][]float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	head := []string{"time_s"}
	for _, tr := range res.Traces {
		head = append(head, fmt.Sprintf("fdtd_r%.2f", float64(tr.Range)))
	}
	if ref != nil {
		for _, tr := range res.Traces {
			head = append(head, fmt.Sprintf("fk_r%.2f", float64(tr.Range)))
		}
	}
	if err := w.Write(head); err != nil {
		return err
	}
	for n := range res.Traces[0].Vertical {
		row := []string{strconv.FormatFloat(res.T0+float64(n)*res.Dt, 'g', 8, 64)}
		for _, tr := range res.Traces {
			row = append(row, strconv.FormatFloat(float64(tr.Vertical[n]), 'g', 8, 64))
		}
		for i := range ref {
			row = append(row, strconv.FormatFloat(ref[i][n], 'g', 8, 64))
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return w.Error()
}

func summarise(res *fdtd.Result, ref [][]float64) {
	fmt.Fprintf(os.Stderr, "\n  %8s  %12s  %12s  %8s  %8s\n",
		"range", "grid peak", "f-k peak", "ratio", "rms")
	for i, tr := range res.Traces {
		var pa, pb, num, den float64
		for n := range tr.Vertical {
			a, b := float64(tr.Vertical[n]), ref[i][n]
			pa = math.Max(pa, math.Abs(a))
			pb = math.Max(pb, math.Abs(b))
			num += (a - b) * (a - b)
			den += b * b
		}
		fmt.Fprintf(os.Stderr, "  %7.2fm  %12.4e  %12.4e  %8.4f  %7.2f%%\n",
			float64(tr.Range), pa, pb, pa/pb, 100*math.Sqrt(num/den))
	}
	fmt.Fprintf(os.Stderr, "\n  the residual is dominated by grid dispersion and falls linearly\n"+
		"  with -spacing; extrapolating two runs to zero spacing is what V5 asserts\n\n")
}
