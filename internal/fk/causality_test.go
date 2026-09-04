package fk

import (
	"math"
	"math/cmplx"
	"testing"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/units"
)

// V9 for layered media: nothing arrives before the medium could carry it.
//
// The homogeneous model has had this check since slice 0, on the impulse
// response rather than on a convolved trace, and for a good reason: a convolved
// trace's precursor is measured against a peak that smoothing has already
// reduced, so the same absolute residue can be made to look like anything.
// V5's waveform agreement does not stand in for it either — two solvers can
// agree closely and both be acausal, and here they share an attenuation model
// deliberately.
//
// The layered case cannot use an impulse response, and finding out why took
// three attempts. A band-limited causal system is not causal: a hard edge at
// Nyquist rings symmetrically in time, and the ringing is a precursor. The
// homogeneous model escapes this because its own attenuation rolls the response
// off inside any reasonable band. A layered stack does not. Measured on this
// medium at 20 m, the response falls to a minimum near 200 Hz and then *rises*
// again — higher modes, which sample the stiff half-space and are far less
// attenuated than the fundamental — and it is still at a sixth of its peak at
// 300 Hz with Q as low as 5. There is no sample rate at which this medium band
// limits its own impulse response.
//
// Two hypotheses were eliminated first, and both mattered. Refining the
// wavenumber quadrature sixty-fourfold moved the response by less than one part
// in a thousand, so it is not integration error. Quadrupling the transform
// window changed the reported precursor not at all — which is the signature of
// a band-limit artefact rather than a wraparound one, since lengthening the
// record at a fixed sample rate does not move the band edge.
//
// So the source provides the band limit instead, which is also how the model is
// used. A Ricker at 20 Hz has nothing left at 120 Hz, so there is no edge to
// ring, and the question asked is the one that matters: does a footstep produce
// ground motion before the fastest wave in the band could have arrived.
func TestLayeredResponseHasNoPrecursor(t *testing.T) {
	const (
		fs   = 400.0
		n    = 4096
		band = 120.0 // where the wavelet has nothing left
		f0   = 20.0
	)
	st := layer.Stack{
		{Thickness: 2, Vp: 350, Vs: 140, Density: 1600, Qs: 20},
		{Vp: 800, Vs: 320, Density: 2000, Qs: 20},
	}
	m := Medium{Stack: st, DefaultQ: 20}
	ranges := []units.Metres{10, 20}
	if err := m.checkLayerPhase(band); err != nil {
		t.Fatalf("the band is outside what the propagator can carry for this stack: %v", err)
	}

	coeff := make([][]complex128, len(ranges))
	for i := range coeff {
		coeff[i] = make([]complex128, n/2+1)
	}
	df := fs / n
	for k := 1; k <= n/2; k++ {
		f := float64(k) * df
		if f > band {
			break
		}
		u, err := m.VerticalDisplacementMulti(ranges, f, Integration{})
		if err != nil {
			t.Fatal(err)
		}
		// A Ricker force, and displacement to velocity.
		fr := f / f0
		mag := 2 / math.Sqrt(math.Pi) * fr * fr * math.Exp(-fr*fr) / f0
		phase := -2 * math.Pi * f * 1.5 / f0
		w := complex(mag, 0) * cmplx.Exp(complex(0, phase))
		for i := range ranges {
			coeff[i][k] = complex(0, 2*math.Pi*f) * u[i] * w * complex(fs, 0)
		}
	}

	// The fastest thing the band can carry: the stiffest layer's compressional
	// velocity, raised by constant-Q dispersion at the top of the band.
	fastest := 0.0
	for _, l := range st {
		v := real(m.complexVelocity(float64(l.Vp), 2.25*m.qOf(l), band))
		fastest = math.Max(fastest, v)
	}
	t.Logf("fastest velocity in the band: %.1f m/s", fastest)

	for i, r := range ranges {
		trace := dsp.IRFFT(coeff[i], n)
		var peak float64
		var peakAt int
		for j, v := range trace {
			if math.Abs(v) > peak {
				peak, peakAt = math.Abs(v), j
			}
		}
		if peak == 0 {
			t.Fatalf("r=%g m: silence", r)
		}

		arrival := int(math.Floor(float64(r) / fastest * fs))
		var early float64
		for j := range max(arrival-1, 0) {
			early = math.Max(early, math.Abs(trace[j])/peak)
		}
		// Negative time is the far end of the circular record. The window is
		// ten seconds for arrivals inside a tenth of one, so anything there is
		// a precursor rather than an undecayed coda.
		var back float64
		for b := 1; b < n/4; b++ {
			back = math.Max(back, math.Abs(trace[n-b])/peak)
		}
		t.Logf("r=%4.0f m: peak at %.4f s, nothing may arrive before %.4f s; "+
			"largest sample before it %.2e of peak, largest at negative time %.2e",
			r, float64(peakAt)/fs, float64(arrival)/fs, early, back)

		// The bound is 1e-4, against measured values of 2e-7 to 2e-6 for the
		// precursor and about 1e-5 at negative time. It matches the homogeneous
		// model's 2e-4 for a measured 6e-5, and for the same reason: a response
		// assembled from propagators and a quadrature carries a broadband
		// residue, and a residue lands at negative time as readily as anywhere
		// else. What the bound rules out is a precursor of the size a real
		// causality failure produces, which is percents.
		if early > 1e-4 {
			t.Errorf("r=%g m: %.2e of peak arrives before the fastest wave in the band could", r, early)
		}
		if back > 1e-4 {
			t.Errorf("r=%g m: %.2e of peak sits at negative time", r, back)
		}
		if peakAt < arrival {
			t.Errorf("r=%g m: the peak precedes the first possible arrival", r)
		}
	}
}
