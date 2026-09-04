// Package hier measures the modelling hierarchy: what each level costs, and
// what it costs you.
//
// The plan sets out four levels — a closed-form far-field model, a layered
// wavenumber integration, a precomputed bank of the same, and a time-domain
// grid — and says they exist so that a run can be repeated with better physics
// and nothing else changed. That is only useful if the difference between them
// is known. Otherwise "better physics" is an assertion, and choosing a level is
// a matter of taste rather than of what a given claim can survive.
//
// So this package puts them on one axis. Every level answers the same question
// — vertical surface velocity per newton of vertical surface force — and is
// scored against the same reference on the same frequency grid, in both the
// frequency domain and as a synthesised trace. The cost of each is measured in
// the same run, because an error figure without a cost figure does not tell
// anyone which level to use.
//
// The reference is the wavenumber integration at heavily increased sampling.
// That is not an independent check of itself: what makes it trustworthy is V5,
// where a time-domain solver sharing none of its code reproduces it to a
// fraction of a percent. This package measures departures from it; slice 4
// establishes that it is the right thing to depart from.
package hier

import (
	"fmt"
	"math"
	"math/cmplx"
	"time"

	"geosim.dev/geosim/internal/bank"
	"geosim.dev/geosim/internal/fk"
	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// Level is one rung: something that can produce a vertical surface velocity
// response, and a record of what it cost to get into a position to do so.
type Level struct {
	// Name is how the level appears in a table.
	Name string
	// Where says what part of the system can afford it, which is the column
	// readers actually act on.
	Where string
	// Response is vertical surface velocity per unit vertical surface point
	// force, in (m/s)/N, for every range at one frequency. Every level returns
	// the same quantity in the same units; a level that returned displacement
	// would look like a different physics rather than a different unit, which
	// is why the conversion belongs inside each level and not in the
	// comparison.
	//
	// Batched over ranges rather than called per range, because for the
	// wavenumber level that is not an optimisation but the difference between
	// a comparison that runs and one that does not: the expensive part of the
	// solve does not depend on range at all, only the Hankel weight does, so
	// calling it per range repeats every solve once per receiver.
	Response func(ranges []units.Metres, f float64) ([]complex128, error)
	// Setup is the one-off cost paid before the first Response call — nil for
	// the closed forms, minutes for a bank.
	Setup time.Duration
	// Note records anything about the level that a number cannot carry.
	Note string
}

// Options configures a comparison.
type Options struct {
	// Ranges to score at.
	Ranges []units.Metres
	// Band is the highest frequency modelled.
	Band float64
	// BandLow is the lowest, and it is not a detail. A level is scored over
	// the band it is asked to work in, and the far-field model in particular
	// is asymptotic in kr — at any fixed range the bottom of a wide band is
	// deep in the near field, so a median taken from zero mostly measures the
	// model where it does not claim to work. A footstep's energy is in the
	// tens of hertz; scoring from there says something about the model, and
	// scoring from a fraction of a hertz says something about the band.
	// Zero starts at the first bin.
	BandLow float64
	// Rate and Samples set the frequency grid, which is shared by every level
	// so that a bank can be scored on the bins it actually stores rather than
	// on an interpolation of them.
	Rate    float64
	Samples int
	// Q is the quality factor for layers that do not carry one.
	Q float64
	// WaveletFreq is the peak frequency of the Ricker used for the trace
	// comparison. Zero uses a fifth of the band.
	WaveletFreq float64
	// ReferenceRefinement multiplies the reference's wavenumber sampling.
	// Zero uses 8.
	ReferenceRefinement int
}

func (o Options) waveletFreq() float64 {
	if o.WaveletFreq > 0 {
		return o.WaveletFreq
	}
	return o.Band / 5
}

func (o Options) refinement() int {
	if o.ReferenceRefinement > 0 {
		return o.ReferenceRefinement
	}
	return 8
}

func (o Options) validate() error {
	switch {
	case len(o.Ranges) == 0:
		return fmt.Errorf("hier: no ranges given")
	case o.Band <= 0:
		return fmt.Errorf("hier: band must be positive, got %g Hz", o.Band)
	case o.BandLow < 0 || o.BandLow >= o.Band:
		return fmt.Errorf("hier: band low %g Hz is not below the band %g Hz", o.BandLow, o.Band)
	case o.Rate <= 0 || o.Samples <= 0:
		return fmt.Errorf("hier: the frequency grid needs a positive rate and length")
	case o.Band >= o.Rate/2:
		return fmt.Errorf("hier: band %g Hz is at or above Nyquist for %g Hz", o.Band, o.Rate)
	}
	return nil
}

// bins are the grid indices inside the band, skipping DC.
func (o Options) bins() []int {
	df := o.Rate / float64(o.Samples)
	var out []int
	for k := 1; k <= o.Samples/2; k++ {
		f := float64(k) * df
		if f > o.Band {
			break
		}
		if f < o.BandLow {
			continue
		}
		out = append(out, k)
	}
	return out
}

// Analytic is the closed-form far-field surface-wave model of slice 0, applied
// to the surface layer.
//
// Applied to the *surface* layer specifically, because that is the mistake the
// level represents. Someone with a layered site and a closed-form model reaches
// for the material they can see, and the question this level answers is what
// that costs — not what the best possible half-space approximation would cost,
// which nobody has in the field.
func Analytic(st layer.Stack, q float64) Level {
	top := st[0]
	if q <= 0 {
		q = 30
	}
	if top.Qs > 0 {
		q = top.Qs
	}
	g := green.HalfSpaceGF{Soil: soil.HalfSpace{
		Vp: top.Vp, Vs: top.Vs, Density: top.Density, Qs: q,
	}}
	return Level{
		Name:  "L0 analytic",
		Where: "runtime, any range",
		Response: func(ranges []units.Metres, f float64) ([]complex128, error) {
			out := make([]complex128, len(ranges))
			for i, r := range ranges {
				v, err := g.VelocityResponse(r, units.Hertz(f))
				if err != nil {
					return nil, err
				}
				out[i] = v
			}
			return out, nil
		},
		Note: "far-field surface wave in the top layer; no layering, no near field",
	}
}

// Wavenumber is the layered integration, at a stated multiple of the sampling
// the solver would choose for itself.
func Wavenumber(st layer.Stack, q float64, refine int, name, where string) (Level, error) {
	m := fk.Medium{Stack: st, DefaultQ: q}
	return Level{
		Name:  name,
		Where: where,
		Response: func(ranges []units.Metres, f float64) ([]complex128, error) {
			opt := fk.Integration{}
			if refine > 1 {
				g, err := m.GridFor(ranges, f, fk.Integration{})
				if err != nil {
					return nil, err
				}
				opt.Samples = refine * g.Samples
			}
			u, err := m.VerticalDisplacementMulti(ranges, f, opt)
			if err != nil {
				return nil, err
			}
			// Displacement to velocity, so that every level is scored on the
			// quantity a geophone sees.
			jw := complex(0, 2*math.Pi*f)
			for i := range u {
				u[i] *= jw
			}
			return u, nil
		},
		Note: fmt.Sprintf("wavenumber sampling x%d", max(refine, 1)),
	}, nil
}

// Banked builds a bank over the ranges and returns a level that interpolates
// it, with the build charged to Setup.
//
// The range grid is the coarsest that still satisfies the unwrapping condition
// of internal/bank, which is the grid a real bank would be built on: making it
// finer would flatter the level by removing the interpolation error that is the
// whole reason a bank is cheap.
func Banked(st layer.Stack, opt Options) (Level, *bank.Bank, error) {
	start := time.Now()
	m := fk.Medium{Stack: st, DefaultQ: opt.Q}
	slowest, _ := st.VelocityBounds()
	step := bank.RangeNyquist(0.87*float64(slowest), opt.Band) / 2

	rMin, rMax := math.Inf(1), 0.0
	for _, r := range opt.Ranges {
		rMin = math.Min(rMin, float64(r))
		rMax = math.Max(rMax, float64(r))
	}
	// Offset the grid by half a step so that no scored range lands on a node.
	// A bank is exact at its nodes, and interpolation error is the whole of
	// what this level is being scored for, so scoring it where that error does
	// not exist would report a bank as free of the one cost it has.
	rMin -= step / 2
	rMax += step / 2
	count := int(math.Ceil((rMax-rMin)/step)) + 1
	if count < 2 {
		count = 2
	}
	h := bank.Header{
		FormatVersion: bank.FormatVersion,
		Provenance:    bank.Provenance{Solver: "geosim fk", Notes: "hierarchy comparison"},
		Medium:        st,
		SampleRateHz:  opt.Rate,
		Samples:       opt.Samples,
		Ranges:        bank.RangeGrid{MinM: rMin, MaxM: rMax, Count: count},
		Component:     "vertical surface displacement per unit vertical surface point force",
		Units:         "m/N",
	}
	if err := h.Validate(); err != nil {
		return Level{}, nil, err
	}
	if err := h.CheckRangeSampling(opt.Band); err != nil {
		return Level{}, nil, err
	}
	b, err := bank.New(h)
	if err != nil {
		return Level{}, nil, err
	}
	ranges := make([]units.Metres, count)
	for i := range ranges {
		ranges[i] = units.Metres(h.Ranges.At(i))
	}
	for _, k := range opt.bins() {
		u, err := m.VerticalDisplacementMulti(ranges, h.FrequencyAt(k), fk.Integration{})
		if err != nil {
			return Level{}, nil, err
		}
		for i := range ranges {
			if err := b.Set(i, k, u[i]); err != nil {
				return Level{}, nil, err
			}
		}
	}
	df := opt.Rate / float64(opt.Samples)
	return Level{
		Name:  "L1b bank",
		Where: "runtime, precomputed",
		Setup: time.Since(start),
		Response: func(ranges []units.Metres, f float64) ([]complex128, error) {
			k := int(math.Round(f / df))
			if math.Abs(float64(k)*df-f) > 1e-9*df {
				return nil, fmt.Errorf("hier: %g Hz is not on the bank's grid", f)
			}
			out := make([]complex128, len(ranges))
			jw := complex(0, 2*math.Pi*f)
			for i, r := range ranges {
				u, err := b.Response(r, k)
				if err != nil {
					return nil, err
				}
				out[i] = jw * u
			}
			return out, nil
		},
		Note: fmt.Sprintf("%d ranges at %.3f m, %s", count, h.Ranges.Spacing(), sizeOf(b.SizeBytes())),
	}, b, nil
}

func sizeOf(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f kB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

// Score is one level at one range.
type Score struct {
	Level string
	Range units.Metres
	// SpectralMedian and SpectralMax are relative departures from the
	// reference across the band. The median says what the level is typically
	// worth; the maximum says where it breaks, and the two are far apart for
	// exactly the levels whose failure is confined to part of the band.
	SpectralMedian, SpectralMax float64
	// WorstAt is the frequency of the maximum.
	WorstAt float64
	// PeakRatio and TraceRMS compare synthesised traces, which is the form the
	// error actually reaches a detector in.
	PeakRatio, TraceRMS float64
	// PerResponse is the mean cost of one range-frequency pair, with the level
	// answering every range at once. For the wavenumber level that sharing is
	// most of why a bank is affordable to build, so charging it the unshared
	// cost would misprice the thing the table exists to price.
	PerResponse time.Duration
}

// Report is a whole comparison.
type Report struct {
	Reference string
	Options   Options
	Levels    []Level
	Scores    []Score
}

// Compare scores every level against the reference.
func Compare(ref Level, levels []Level, opt Options) (*Report, error) {
	if err := opt.validate(); err != nil {
		return nil, err
	}
	bins := opt.bins()
	if len(bins) < 8 {
		return nil, fmt.Errorf("hier: only %d bins inside the band; widen it or lengthen the transform", len(bins))
	}

	// The reference spectra and traces, once.
	refSpec, _, err := spectra(ref, bins, opt)
	if err != nil {
		return nil, fmt.Errorf("hier: reference: %w", err)
	}
	refTrace := make([][]float64, len(opt.Ranges))
	for i := range opt.Ranges {
		refTrace[i] = synthesise(refSpec[i], bins, opt)
	}

	rep := &Report{Reference: ref.Name, Options: opt, Levels: levels}
	for _, lv := range levels {
		spec, per, err := spectra(lv, bins, opt)
		if err != nil {
			return nil, fmt.Errorf("hier: %s: %w", lv.Name, err)
		}
		if per, err = refineTiming(lv, bins, opt, per); err != nil {
			return nil, fmt.Errorf("hier: timing %s: %w", lv.Name, err)
		}
		for i, r := range opt.Ranges {
			tr := synthesise(spec[i], bins, opt)
			rep.Scores = append(rep.Scores,
				score(lv.Name, r, spec[i], refSpec[i], tr, refTrace[i], bins, opt, per))
		}
	}
	return rep, nil
}

// spectra evaluates one level over the whole grid, returning the spectrum per
// range and the mean cost of one range-frequency pair.
func spectra(lv Level, bins []int, opt Options) ([][]complex128, time.Duration, error) {
	df := opt.Rate / float64(opt.Samples)
	out := make([][]complex128, len(opt.Ranges))
	for i := range out {
		out[i] = make([]complex128, len(bins))
	}
	start := time.Now()
	for j, k := range bins {
		v, err := lv.Response(opt.Ranges, float64(k)*df)
		if err != nil {
			return nil, 0, err
		}
		for i := range opt.Ranges {
			out[i][j] = v[i]
		}
	}
	per := time.Since(start) / time.Duration(len(bins)*len(opt.Ranges))
	return out, per, nil
}

// refineTiming re-measures a level that finished too quickly to time.
//
// A bank lookup is tens of nanoseconds and the whole sweep is a millisecond,
// which on a clock with microsecond resolution reports as a suspiciously round
// number rather than as a measurement. Repeating until the total is long enough
// to mean something turns the cheapest levels — the ones the table exists to
// recommend — back into measurements.
const timingFloor = 50 * time.Millisecond

func refineTiming(lv Level, bins []int, opt Options, first time.Duration) (time.Duration, error) {
	pairs := time.Duration(len(bins) * len(opt.Ranges))
	if first*pairs >= timingFloor {
		return first, nil
	}
	df := opt.Rate / float64(opt.Samples)
	start := time.Now()
	reps := 0
	for time.Since(start) < timingFloor {
		for _, k := range bins {
			if _, err := lv.Response(opt.Ranges, float64(k)*df); err != nil {
				return 0, err
			}
		}
		reps++
	}
	return time.Since(start) / (time.Duration(reps) * pairs), nil
}

func score(name string, r units.Metres, got, want []complex128, gotTr, wantTr []float64,
	bins []int, opt Options, per time.Duration) Score {
	df := opt.Rate / float64(opt.Samples)
	s := Score{Level: name, Range: r, PerResponse: per}
	rel := make([]float64, len(got))
	for i := range got {
		den := cmplx.Abs(want[i])
		if den == 0 {
			continue
		}
		rel[i] = cmplx.Abs(got[i]-want[i]) / den
		if rel[i] > s.SpectralMax {
			s.SpectralMax = rel[i]
			s.WorstAt = float64(bins[i]) * df
		}
	}
	s.SpectralMedian = median(rel)

	var pa, pb, num, den float64
	for i := range wantTr {
		pa = math.Max(pa, math.Abs(gotTr[i]))
		pb = math.Max(pb, math.Abs(wantTr[i]))
		num += (gotTr[i] - wantTr[i]) * (gotTr[i] - wantTr[i])
		den += wantTr[i] * wantTr[i]
	}
	if pb > 0 {
		s.PeakRatio = pa / pb
	}
	if den > 0 {
		s.TraceRMS = math.Sqrt(num / den)
	}
	return s
}

func median(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	c := append([]float64(nil), x...)
	for i := 1; i < len(c); i++ {
		for j := i; j > 0 && c[j] < c[j-1]; j-- {
			c[j], c[j-1] = c[j-1], c[j]
		}
	}
	return c[len(c)/2]
}
