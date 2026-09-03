package bank

import (
	"fmt"
	"math"
	"math/cmplx"
	"sync"

	"geosim.dev/geosim/internal/units"
)

// Response returns the Green's function at an arbitrary range and frequency
// bin, interpolated between the two bracketing grid ranges.
//
// Interpolation is linear in log-magnitude and in unwrapped phase, separately.
// Both quantities are very nearly linear in range — phase because it is
// essentially -k*r for whatever k the medium gives at that frequency,
// log-magnitude because it is the sum of a slowly varying geometric term and
// an attenuation term proportional to range — so linear interpolation of the
// pair is accurate on a grid far coarser than either the wavelength or the
// sampling interval.
//
// Interpolating the complex values directly would not work. Two responses at
// ranges differing by a fraction of a wavelength are of similar magnitude and
// nearly opposite phase; their average has a magnitude near zero. In the time
// domain the same statement is that averaging two arrivals at different times
// gives a waveform with two peaks where the true one has a single arrival
// in between.
func (b *Bank) Response(r units.Metres, bin int) (complex128, error) {
	if bin < 0 || bin >= b.Bins() {
		return 0, fmt.Errorf("bank: bin %d out of range [0, %d)", bin, b.Bins())
	}
	lo, frac, err := b.bracket(float64(r))
	if err != nil {
		return 0, err
	}
	b.prepare()

	i0 := lo*b.Bins() + bin
	i1 := (lo+1)*b.Bins() + bin
	if b.Ranges.Count == 1 {
		i1 = i0
	}
	// A bin that is identically zero in the bank — above the modelled band,
	// or at DC — stays zero rather than becoming exp(-inf).
	if b.logMag[i0] == negInf || b.logMag[i1] == negInf {
		return 0, nil
	}
	logMag := b.logMag[i0] + frac*(b.logMag[i1]-b.logMag[i0])
	phase := b.phase[i0] + frac*(b.phase[i1]-b.phase[i0])
	return cmplx.Rect(math.Exp(logMag), phase), nil
}

const negInf = -1e308

// prepare builds the log-magnitude and range-unwrapped phase tables the
// interpolation reads. Done once, lazily, because a bank that is only ever
// queried on its own grid points does not need them.
//
// The unwrapping is along range, not frequency. Walking the range axis at a
// fixed frequency, the phase advances by about -omega*dr/c per step, and the
// principal value that cmplx.Phase returns has lost however many whole cycles
// that is. Accumulating the wrapped differences recovers them — provided each
// step is under half a cycle, which is exactly the condition RangeNyquist
// enforces at build time.
func (b *Bank) prepare() {
	b.once.Do(func() {
		n := b.Ranges.Count * b.Bins()
		b.logMag = make([]float64, n)
		b.phase = make([]float64, n)

		for k := range b.Bins() {
			var prevWrapped, unwrapped float64
			for i := range b.Ranges.Count {
				j := 2 * (i*b.Bins() + k)
				v := complex(float64(b.data[j]), float64(b.data[j+1]))
				idx := i*b.Bins() + k

				mag := cmplx.Abs(v)
				if mag == 0 {
					b.logMag[idx] = negInf
					b.phase[idx] = unwrapped
					continue
				}
				b.logMag[idx] = math.Log(mag)

				w := cmplx.Phase(v)
				if i == 0 {
					unwrapped = w
				} else {
					d := w - prevWrapped
					// Fold the step into (-pi, pi]; anything larger is a lost
					// cycle rather than a real jump.
					d -= 2 * math.Pi * math.Round(d/(2*math.Pi))
					unwrapped += d
				}
				prevWrapped = w
				b.phase[idx] = unwrapped
			}
		}
	})
}

// RangeNyquist is the coarsest range spacing at which the phase can still be
// unwrapped, for a medium whose slowest phase velocity is slowestVelocity and
// content up to maxFreq.
//
//	dr < c / (2 * f)
//
// It is a sampling condition on the range axis, exactly analogous to the one on
// the time axis, and for the same reason: between two grid ranges the phase
// must advance by less than half a cycle or there is no way to tell how many
// whole cycles were lost.
//
// It is also what sets the size of a bank. At 2 kHz sampling the limit would be
// about 7 cm, which over tens of metres is thousands of ranges; restricting the
// modelled band to a few hundred hertz — where a footstep's energy actually is,
// after attenuation — relaxes it to tens of centimetres and shrinks the bank by
// an order of magnitude.
func RangeNyquist(slowestVelocity, maxFreq float64) float64 {
	if maxFreq <= 0 {
		return math.Inf(1)
	}
	return slowestVelocity / (2 * maxFreq)
}

// CheckRangeSampling reports whether the grid is fine enough to interpolate,
// given the medium and the highest frequency the bank actually carries.
//
// A safety factor of two below the strict limit, because the strict limit is
// where unwrapping becomes ambiguous rather than where it becomes inaccurate.
func (h Header) CheckRangeSampling(maxFreq float64) error {
	if h.Ranges.Count <= 1 || len(h.Medium) == 0 {
		return nil
	}
	slowest, _ := h.Medium.VelocityBounds()
	// Rayleigh runs slower than shear, by about 8% at worst.
	limit := RangeNyquist(0.87*float64(slowest), maxFreq) / 2
	if got := h.Ranges.Spacing(); got > limit {
		return fmt.Errorf("bank: range spacing %.4f m is too coarse to unwrap phase up to %g Hz; need under %.4f m",
			got, maxFreq, limit)
	}
	return nil
}

// interpolation tables, built once by prepare.
type interpTables struct {
	once   sync.Once
	logMag []float64
	phase  []float64
}
