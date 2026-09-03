// Package units carries the SI quantities that flow through the forward
// model, as named types.
//
// The pipeline turns a force into a ground velocity into a voltage, and along
// the way it is easy to hand a displacement to something expecting a velocity,
// or millivolts to something expecting volts. Both mistakes typecheck if
// everything is a float64, and neither shows up as anything more obvious than
// a waveform of the wrong amplitude. So the quantities that actually get
// confused get a type, and it is spent at API boundaries — struct fields and
// function signatures — rather than inside numeric loops, where the
// conversions would cost more clarity than they buy.
//
// Everything is SI. Conversion happens at I/O and nowhere else.
package units

// Time and space.
type (
	// Seconds is a duration or an instant on the simulation clock.
	Seconds float64
	// Metres is a distance or a coordinate.
	Metres float64
	// Hertz is a frequency.
	Hertz float64
	// RadPerSec is an angular frequency.
	RadPerSec float64
)

// Mechanics. Displacement, Velocity and Acceleration are the three that the
// seismic literature moves between constantly and that a geophone sits in the
// middle of: it measures ground velocity and reports volts.
type (
	// Newtons is a force — a ground reaction force, or a contact force.
	Newtons float64
	// Displacement is ground motion, in metres.
	Displacement float64
	// Velocity is ground particle velocity, in metres per second.
	Velocity float64
	// Acceleration is ground particle acceleration, in metres per second squared.
	Acceleration float64
	// Kilograms is a mass.
	Kilograms float64
)

// Medium.
type (
	// SpeedMPS is a wave speed — P, S or Rayleigh — in metres per second.
	SpeedMPS float64
	// DensityKgM3 is a mass density in kilograms per cubic metre.
	DensityKgM3 float64
	// Pascals is a modulus or a stress.
	Pascals float64
)

// Electrical.
type (
	// Volts is a sensor output.
	Volts float64
	// Ohms is a resistance — coil, or shunt.
	Ohms float64
	// VoltsPerMPS is a geophone's velocity sensitivity.
	VoltsPerMPS float64
)

// Physical constants.
const (
	// Boltzmann is Boltzmann's constant, J/K.
	Boltzmann = 1.380649e-23
	// GravityMPS2 is standard gravity, m/s^2.
	GravityMPS2 = 9.80665
	// RoomTemperatureK is 300 K, the reference for Johnson noise here.
	RoomTemperatureK = 300.0
)

// AngularFrequency converts a frequency to an angular frequency.
func AngularFrequency(f Hertz) RadPerSec { return RadPerSec(2 * 3.141592653589793 * float64(f)) }
