package sensing

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/bank"
	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/units"
)

// Propagation is a medium's vertical-velocity response at a range: the ground
// velocity per newton of vertical surface force, as a function of frequency.
//
// An interface so the engine does not care whether the response is the
// homogeneous far-field closed form or a precomputed bank of layered
// wavenumber integrals. The two differ enormously at short range — by a factor
// of three inside a tenth of a wavelength — but they answer the same question,
// and having the engine take either is what lets a run be repeated with better
// physics and nothing else changed.
type Propagation interface {
	// VerticalVelocityResponse is (m/s)/N at range r and frequency f.
	VerticalVelocityResponse(r units.Metres, f units.Hertz) (complex128, error)
	// RadialForceResponse is the vertical velocity per newton of horizontal
	// force directed along the source-receiver line. Implementations that do
	// not model it return zero, which is defensible here: the shear is a
	// smooth half-cycle with no impact transient, and slice 2 measured its
	// contribution at under a third of a percent even head-on.
	RadialForceResponse(r units.Metres, f units.Hertz) (complex128, error)
	// TransformSize is the transform length the model's frequency grid
	// assumes, or zero to let the caller choose. A bank defines its own grid
	// and must be inverted on it.
	TransformSize() int
	// Describe names the model, for provenance in a trace's sidecar.
	Describe() string
}

// analytic is slice 0's homogeneous half-space in far-field form.
type analytic struct{ gf green.HalfSpaceGF }

// Analytic wraps the closed-form homogeneous model.
func Analytic(gf green.HalfSpaceGF) Propagation { return analytic{gf} }

func (a analytic) VerticalVelocityResponse(r units.Metres, f units.Hertz) (complex128, error) {
	return a.gf.VelocityResponse(r, f)
}

func (a analytic) RadialForceResponse(r units.Metres, f units.Hertz) (complex128, error) {
	return a.gf.RadialForceResponse(r, f)
}

func (a analytic) TransformSize() int { return 0 }

func (a analytic) Describe() string {
	return "homogeneous half-space, far-field Rayleigh (" + a.gf.Soil.String() + ")"
}

// banked serves a precomputed bank of layered wavenumber integrals.
//
// The bank stores displacement per force; the sensor measures velocity, so the
// conversion by i*omega happens here rather than being left for a caller to
// remember.
//
// The shear response is deliberately not served. A bank holds one component,
// and mixing a near-field-correct vertical response with a far-field analytic
// horizontal one would be incoherent — the whole reason for the bank is that
// the far-field form is wrong at these ranges. Omitting the shear entirely is
// the honest simplification, and slice 2's sensitivity sweep is what licenses
// it: varying the fore-aft peak over a factor of twenty moved a walk-past by
// less than a percent, because the shear has no impact transient and radiates
// almost nothing in the band that matters.
type banked struct {
	b *bank.Bank
}

// FromBank serves propagation from a precomputed Green's function bank.
func FromBank(b *bank.Bank) (Propagation, error) {
	if b == nil {
		return nil, fmt.Errorf("sensing: bank is nil")
	}
	return banked{b}, nil
}

func (bk banked) VerticalVelocityResponse(r units.Metres, f units.Hertz) (complex128, error) {
	if f <= 0 {
		return 0, nil
	}
	// The bank's frequency grid is the transform it will be inverted with, so
	// a frequency maps to a bin exactly rather than by search.
	x := float64(f) / bk.b.FrequencyAt(1)
	bin := int(math.Round(x))
	if bin < 0 || bin >= bk.b.Bins() {
		return 0, nil
	}
	if math.Abs(x-float64(bin)) > 1e-6 {
		return 0, fmt.Errorf("sensing: %g Hz does not land on the bank's frequency grid (spacing %g Hz)",
			f, bk.b.FrequencyAt(1))
	}
	u, err := bk.b.Response(r, bin)
	if err != nil {
		return 0, err
	}
	// Displacement to velocity.
	return complex(0, 2*math.Pi*float64(f)) * u, nil
}

func (bk banked) RadialForceResponse(units.Metres, units.Hertz) (complex128, error) {
	return 0, nil
}

func (bk banked) TransformSize() int { return bk.b.Samples }

func (bk banked) Describe() string {
	return fmt.Sprintf("bank: %s, %d ranges %g-%g m, %s",
		bk.b.Provenance.Solver, bk.b.Ranges.Count, bk.b.Ranges.MinM, bk.b.Ranges.MaxM, bk.b.Component)
}
