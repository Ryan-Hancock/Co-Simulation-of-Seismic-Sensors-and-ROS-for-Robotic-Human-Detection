package hier

import (
	"math"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/fdtd"
	"geosim.dev/geosim/internal/units"
)

// comparisonForce is the wavelet's peak amplitude. A constant scale cancels out
// of every metric here, so it is chosen to keep the numbers readable rather
// than to represent anything.
const comparisonForce = units.Newtons(1000)

// synthesise turns a level's spectrum into the trace it would produce.
//
// The spectral scores say where a level departs; this says how much of that
// departure survives into a waveform. They are not the same question. A level
// can be badly wrong at frequencies the source barely excites, or where
// attenuation has already removed the signal, and be perfectly usable — which
// is exactly the case for the far-field model at the top of the band. Reporting
// only the spectral error would condemn it for an error nobody can measure.
//
// The wavelet is the same Ricker the V5 comparison uses, so a level's trace
// error here is directly comparable with the grid's there.
func synthesise(spec []complex128, bins []int, opt Options) []float64 {
	coeff := make([]complex128, opt.Samples/2+1)
	df := opt.Rate / float64(opt.Samples)
	f0 := units.Hertz(opt.waveletFreq())
	for i, k := range bins {
		f := float64(k) * df
		coeff[k] = spec[i] * fdtd.RickerSpectrum(comparisonForce, f0, f) * complex(opt.Rate, 0)
	}
	return dsp.IRFFT(coeff, opt.Samples)
}

// SuggestGrid picks a frequency grid for a band: a sample rate comfortably
// above Nyquist and a transform long enough that the response has decayed
// inside it.
//
// Long enough matters more than it looks. The inverse transform is periodic, so
// a response still ringing at the end of the window wraps round to the start,
// and the wrapped tail lands on top of the arrival — where it is
// indistinguishable from a level getting the arrival wrong.
func SuggestGrid(band float64, slowest units.SpeedMPS, furthest units.Metres) (rate float64, samples int) {
	rate = 4 * band
	// Room for the slowest arrival, its coda, and the same again.
	span := 3 * float64(furthest) / (0.87 * float64(slowest))
	samples = dsp.NextPow2(int(math.Ceil(span * rate)))
	if samples < 1024 {
		samples = 1024
	}
	return rate, samples
}
