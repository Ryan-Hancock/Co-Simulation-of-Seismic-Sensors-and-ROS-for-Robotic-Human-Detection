// Package geophone is the last stage of the forward model: ground velocity in,
// volts out.
//
// A moving-coil geophone is a velocity transducer with a mechanical resonance.
// Above its natural frequency it reports ground velocity with a flat response;
// below it, output falls at 12 dB per octave and the phase leads. For footstep
// seismology that corner sits inside the band of interest, so the sensor is not
// a passive observer of the signal — it shapes it, and the model has to say so.
//
// Three stages are modelled separately, because they fail separately and, in
// this project, they are uncertain to very different degrees:
//
//   - Ground coupling. How faithfully the case follows the ground. On a robot
//     this is the least known parameter in the whole chain (O3), so it is a
//     first-class configurable block rather than an assumed unity.
//   - The transducer. Well characterised by a datasheet, and effectively exact.
//   - Noise. Johnson noise of the coil and shunt, which sets the sensor's own
//     floor — as distinct from the site's, which is usually far higher.
package geophone

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/units"
)

// Geophone is a moving-coil vertical geophone and its load.
//
// Field values come from a datasheet and belong in config, never in code: WP4
// may target a different sensor, and O4's domain randomisation needs to move
// them. The zero value is not useful; use SM24 or build one explicitly.
type Geophone struct {
	// NaturalFreq is the undamped natural frequency f0.
	NaturalFreq units.Hertz
	// Sensitivity is the open-circuit transduction constant G.
	Sensitivity units.VoltsPerMPS
	// CoilResistance is the coil's DC resistance Rc.
	CoilResistance units.Ohms
	// ShuntResistance is the load across the coil. Zero or negative means
	// open circuit — no shunt damping, no signal division.
	ShuntResistance units.Ohms
	// OpenCircuitDamping is the mechanical damping ratio with no load.
	OpenCircuitDamping float64
	// MovingMass is the mass of the coil assembly, needed to convert a shunt
	// resistance into a damping ratio.
	MovingMass units.Kilograms
	// Coupling describes how the case follows the ground.
	Coupling Coupling
	// Temperature is the coil temperature for Johnson noise, in kelvin. Zero
	// means units.RoomTemperatureK.
	Temperature float64
}

// SM24 is a Geosense/ION SM-24, the reference part for this project, with an
// open circuit and a well-planted spike.
//
// The values are the nominal datasheet ones. Real units vary by a few percent
// on sensitivity and coil resistance, which is a domain-randomisation axis for
// O4 rather than something to average away here.
func SM24() Geophone {
	return Geophone{
		NaturalFreq:        4.5,
		Sensitivity:        28.8,
		CoilResistance:     375,
		ShuntResistance:    0,
		OpenCircuitDamping: 0.34,
		MovingMass:         0.011,
		Coupling:           WellPlantedSpike(),
		Temperature:        units.RoomTemperatureK,
	}
}

// Coupling is the transfer from ground motion to case motion, modelled as the
// second-order system a mass on a compliant contact actually is: unity at DC,
// a resonance where the contact stiffness and the sensor mass meet, and
// decoupling above it.
//
// Krohn's classic result is that a well-planted spike in firm soil resonates
// far above the seismic band and is effectively transparent, while a sensor
// sitting on soft ground can resonate low enough to sit inside it. A geophone
// carried by a robot is nearer the second case than the first, and the
// resonance moves with soil, load and mounting. That variability is the point:
// it is O3's problem, and it must be parameterised to be studied.
type Coupling struct {
	// ResonanceFreq is the coupling resonance. Zero means perfect coupling.
	ResonanceFreq units.Hertz
	// Damping is its damping ratio.
	Damping float64
}

// WellPlantedSpike is a spike driven into firm soil: a resonance well above
// the footstep band, so nearly transparent below 100 Hz.
func WellPlantedSpike() Coupling { return Coupling{ResonanceFreq: 200, Damping: 0.5} }

// PerfectCoupling is the idealisation — the case follows the ground exactly.
// Useful for isolating the transducer in tests; not a physical claim.
func PerfectCoupling() Coupling { return Coupling{} }

// Validate reports whether the parameters describe a usable sensor.
func (g Geophone) Validate() error {
	switch {
	case g.NaturalFreq <= 0:
		return fmt.Errorf("geophone: natural frequency must be positive, got %g Hz", g.NaturalFreq)
	case g.Sensitivity <= 0:
		return fmt.Errorf("geophone: sensitivity must be positive, got %g V/(m/s)", g.Sensitivity)
	case g.CoilResistance <= 0:
		return fmt.Errorf("geophone: coil resistance must be positive, got %g ohm", g.CoilResistance)
	case g.MovingMass <= 0:
		return fmt.Errorf("geophone: moving mass must be positive, got %g kg", g.MovingMass)
	case g.OpenCircuitDamping <= 0:
		return fmt.Errorf("geophone: open-circuit damping must be positive, got %g", g.OpenCircuitDamping)
	case g.Coupling.ResonanceFreq < 0:
		return fmt.Errorf("geophone: coupling resonance must not be negative, got %g Hz", g.Coupling.ResonanceFreq)
	}
	return nil
}

// omega0 is the undamped natural angular frequency.
func (g Geophone) omega0() float64 { return 2 * math.Pi * float64(g.NaturalFreq) }

// Damping is the total damping ratio: mechanical, plus the electrical damping
// the shunt introduces by letting the coil drive a current.
//
//	zeta = zeta_oc + G^2 / (2 * w0 * M * (Rc + Rs))
//
// The load is therefore part of the sensor model, not part of the DAQ. Halving
// the shunt changes the response shape, and a model that fixes damping at the
// datasheet's open-circuit figure while the field unit runs a damping resistor
// will mispredict amplitude and phase across the whole low-frequency band.
func (g Geophone) Damping() float64 {
	if g.ShuntResistance <= 0 {
		return g.OpenCircuitDamping
	}
	gs := float64(g.Sensitivity)
	total := float64(g.CoilResistance + g.ShuntResistance)
	return g.OpenCircuitDamping + gs*gs/(2*g.omega0()*float64(g.MovingMass)*total)
}

// ShuntForDamping returns the load resistance giving a total damping ratio of
// want, or an error if no positive resistance can reach it — a shunt can only
// add damping, so anything at or below the open-circuit value is unreachable.
func (g Geophone) ShuntForDamping(want float64) (units.Ohms, error) {
	if want <= g.OpenCircuitDamping {
		return 0, fmt.Errorf("geophone: damping %g is not above the open-circuit value %g; a shunt can only add damping", want, g.OpenCircuitDamping)
	}
	gs := float64(g.Sensitivity)
	total := gs * gs / (2 * g.omega0() * float64(g.MovingMass) * (want - g.OpenCircuitDamping))
	rs := total - float64(g.CoilResistance)
	if rs <= 0 {
		return 0, fmt.Errorf("geophone: damping %g needs a load of %g ohm, below the coil's own %g ohm", want, total, g.CoilResistance)
	}
	return units.Ohms(rs), nil
}

// signalDivision is the fraction of the coil's EMF that appears across the
// shunt. Open circuit is unity.
func (g Geophone) signalDivision() float64 {
	if g.ShuntResistance <= 0 {
		return 1
	}
	return float64(g.ShuntResistance) / float64(g.CoilResistance+g.ShuntResistance)
}

// TransducerResponse is the coil's own response at frequency f, in
// V/(m/s) of case velocity, including the shunt's signal division:
//
//	H(s) = G * s^2 / (s^2 + 2*zeta*w0*s + w0^2)
//
// Flat and real above f0; magnitude G/(2*zeta) with a 90 degree phase lead at
// f0; falling as f^2 below it.
func (g Geophone) TransducerResponse(f units.Hertz) complex128 {
	w := 2 * math.Pi * float64(f)
	w0 := g.omega0()
	s := complex(0, w)
	num := complex(float64(g.Sensitivity)*g.signalDivision(), 0) * s * s
	den := s*s + complex(2*g.Damping()*w0, 0)*s + complex(w0*w0, 0)
	return num / den
}

// CouplingResponse is the ground-to-case transfer at f, dimensionless:
//
//	H(s) = wc^2 / (s^2 + 2*zeta_c*wc*s + wc^2)
//
// Unity at DC, peaked near the resonance, falling above it as the case stops
// following the ground.
func (g Geophone) CouplingResponse(f units.Hertz) complex128 {
	if g.Coupling.ResonanceFreq <= 0 {
		return 1
	}
	w := 2 * math.Pi * float64(f)
	wc := 2 * math.Pi * float64(g.Coupling.ResonanceFreq)
	s := complex(0, w)
	return complex(wc*wc, 0) / (s*s + complex(2*g.Coupling.Damping*wc, 0)*s + complex(wc*wc, 0))
}

// Response is the whole sensor at f: ground velocity in m/s to volts out.
func (g Geophone) Response(f units.Hertz) complex128 {
	return g.CouplingResponse(f) * g.TransducerResponse(f)
}

// Apply filters a ground-velocity trace, sampled at fs Hz, into volts.
//
// The filter is applied in the frequency domain, which makes it exact for the
// finite record rather than approximate as an IIR realisation would be. That
// matters here because the pole sits at 4.5 Hz, low enough that a discretised
// filter's warping is visible in the band the detector works in.
func (g Geophone) Apply(velocity []float64, fs float64) ([]units.Volts, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	if fs <= 0 {
		return nil, fmt.Errorf("geophone: sample rate must be positive, got %g", fs)
	}
	if len(velocity) == 0 {
		return nil, nil
	}
	// Pad to a power of two so the wrap-around of the circular filter lands in
	// padding rather than in the record. The sensor's impulse response rings
	// for a few cycles of f0, so the padding is sized from that.
	ring := int(math.Ceil(4 * fs / float64(g.NaturalFreq)))
	n := dsp.NextPow2(len(velocity) + ring)

	padded := make([]float64, n)
	copy(padded, velocity)

	coeff := dsp.RFFT(padded)
	for k, f := range dsp.FreqBins(n, fs) {
		coeff[k] *= g.Response(units.Hertz(f))
	}
	out := dsp.IRFFT(coeff, n)

	volts := make([]units.Volts, len(velocity))
	for i := range volts {
		volts[i] = units.Volts(out[i])
	}
	return volts, nil
}

// NoiseDensity is the sensor's own voltage noise, in V/sqrt(Hz).
//
// The coil and its shunt are both resistors at temperature T, and both are
// noisy. Referred to the output the two combine into the Johnson noise of
// their parallel resistance — the coil's noise is divided by the same factor
// as the signal, the shunt's is divided by the complement, and the algebra
// collapses. So loading a geophone for damping costs a little signal and
// slightly less noise, not the other way round.
//
// This is the *sensor's* floor. It is not the detection floor: ambient ground
// motion at any real site is far above it, which is the point worth carrying
// into O5 — the limit on detection range is the site, not the transducer.
func (g Geophone) NoiseDensity() float64 {
	t := g.Temperature
	if t <= 0 {
		t = units.RoomTemperatureK
	}
	r := float64(g.CoilResistance)
	if g.ShuntResistance > 0 {
		rs := float64(g.ShuntResistance)
		r = r * rs / (r + rs)
	}
	return math.Sqrt(4 * units.Boltzmann * t * r)
}

// NoiseDensityInVelocity is NoiseDensity referred back to ground velocity, in
// (m/s)/sqrt(Hz), at frequencies above f0 where the response is flat. Below f0
// the equivalent input noise rises steeply, which is why footstep detection
// gets no help from the sub-resonance band.
func (g Geophone) NoiseDensityInVelocity() float64 {
	return g.NoiseDensity() / (float64(g.Sensitivity) * g.signalDivision())
}

// Noise returns n samples of the sensor's own voltage noise at sample rate fs.
//
// White with the density NoiseDensity reports: a real coil is not quite white
// once its own resonance is included, but the departure is far below the site
// noise that dominates in practice, and pretending otherwise would be
// precision the model has not earned.
func (g Geophone) Noise(n int, fs float64, rng *rand.Rand) []units.Volts {
	// Discrete white noise of variance s^2 has a one-sided PSD of 2*s^2/fs.
	sigma := g.NoiseDensity() * math.Sqrt(fs/2)
	out := make([]units.Volts, n)
	for i := range out {
		out[i] = units.Volts(rng.NormFloat64() * sigma)
	}
	return out
}

// MagnitudeDB is |Response| in dB relative to the sensitivity, for plotting
// against a datasheet curve.
func (g Geophone) MagnitudeDB(f units.Hertz) float64 {
	return 20 * math.Log10(cmplx.Abs(g.Response(f))/float64(g.Sensitivity))
}

// response is Coupling's own transfer at f, independent of any sensor.
func (c Coupling) response(f units.Hertz) complex128 {
	return Geophone{Coupling: c}.CouplingResponse(f)
}
