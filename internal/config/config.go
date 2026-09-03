// Package config is the run description: everything that determines a
// synthesised trace, in one hashable document.
//
// The hash matters more than it looks. O4's domain randomisation is only
// meaningful if a run replays exactly, and a figure in the eventual validation
// report is only evidence if the run behind it can be reproduced. So every
// trace carries the hash of its fully resolved configuration and the explicit
// RNG seed that went with it, and nothing in the model reads a default that is
// not written back into that resolved form.
//
// JSON rather than YAML, to avoid a dependency for something the standard
// library already does. It is a worse format to hand-write and a better one to
// hash.
package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"geosim.dev/geosim/internal/geophone"
	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/source"
	"geosim.dev/geosim/internal/units"
)

// Config is a complete run description.
type Config struct {
	Soil     SoilSpec   `json:"soil"`
	Walker   WalkerSpec `json:"walker"`
	Sensor   SensorSpec `json:"sensor"`
	Geometry Geometry   `json:"geometry"`
	Walk     WalkSpec   `json:"walk"`
	Sampling Sampling   `json:"sampling"`
	Noise    NoiseSpec  `json:"noise"`
}

// SoilSpec names a preset medium or gives one explicitly. A preset is
// expanded into explicit values before hashing, so changing what a preset
// means changes the hash — which is the intent: the hash identifies the
// physics, not the spelling.
type SoilSpec struct {
	Preset  string  `json:"preset,omitempty"`
	Vp      float64 `json:"vp_mps,omitempty"`
	Vs      float64 `json:"vs_mps,omitempty"`
	Density float64 `json:"density_kgm3,omitempty"`
	Qs      float64 `json:"qs,omitempty"`
}

type WalkerSpec struct {
	Mass            float64 `json:"mass_kg"`
	StanceDuration  float64 `json:"stance_duration_s"`
	FirstPeak       float64 `json:"first_peak_bw"`
	SecondPeak      float64 `json:"second_peak_bw"`
	MidstanceValley float64 `json:"midstance_valley_bw"`
	// TransientPeak and TransientRise describe the heel strike: the height it
	// reaches above the smooth curve, in body weights, and the time it takes
	// to get there. Zero uses the defaults; a negative peak removes it.
	//
	// These are exposed because they are the least constrained part of the
	// whole model and the part the radiated signal depends on most, which
	// makes them the first axes any sensitivity analysis has to sweep.
	TransientPeak float64 `json:"transient_peak_bw,omitempty"`
	TransientRise float64 `json:"transient_rise_s,omitempty"`
	APPeak        float64 `json:"ap_peak_bw,omitempty"`
}

type SensorSpec struct {
	Preset string `json:"preset,omitempty"`
	// TargetDamping, if set, selects the shunt resistance that achieves it.
	// Expressing it this way rather than as a resistance keeps the config in
	// the terms the response is actually shaped by.
	TargetDamping     float64 `json:"target_damping,omitempty"`
	CouplingFreq      float64 `json:"coupling_resonance_hz,omitempty"`
	CouplingDamping   float64 `json:"coupling_damping,omitempty"`
	NaturalFreq       float64 `json:"natural_freq_hz,omitempty"`
	Sensitivity       float64 `json:"sensitivity_v_per_mps,omitempty"`
	CoilResistance    float64 `json:"coil_resistance_ohm,omitempty"`
	MovingMass        float64 `json:"moving_mass_kg,omitempty"`
	OpenCircuitDamped float64 `json:"open_circuit_damping,omitempty"`
}

// Geometry places the walker relative to the sensor.
//
// The sensor sits at the origin and the walker travels along a line offset by
// Range, so Range is the closest approach rather than a fixed separation. That
// is the geometry WP4 will actually record: a walk-past, whose rise and fall
// comes from the changing range to each footfall.
type Geometry struct {
	// Range is the perpendicular distance from the walking line to the
	// sensor, in metres — the closest the walker gets.
	Range float64 `json:"range_m"`
	// ApproachLength is how far along the path the walk starts before, and
	// ends after, closest approach. Zero uses ten metres.
	ApproachLength float64 `json:"approach_length_m,omitempty"`
}

// WalkSpec describes the gait as motion rather than as a single stance.
type WalkSpec struct {
	// Speed in metres per second. Zero uses a comfortable 1.3.
	Speed float64 `json:"speed_mps,omitempty"`
	// Width is the lateral separation between the feet. Zero uses 0.12 m.
	Width float64 `json:"stance_width_m,omitempty"`
}

type Sampling struct {
	Rate float64 `json:"rate_hz"`
	Lead float64 `json:"lead_s"`
	Tail float64 `json:"tail_s"`
}

type NoiseSpec struct {
	Enabled bool   `json:"enabled"`
	Seed    uint64 `json:"seed"`
}

// Default is a runnable configuration: a 75 kg walker at 10 m over loam,
// recorded on an SM-24 shunted to 0.7 of critical at 2 kHz.
func Default() Config {
	return Config{
		Soil:     SoilSpec{Preset: "loam"},
		Walker:   WalkerSpec{Mass: 75, StanceDuration: 0.62, FirstPeak: 1.15, SecondPeak: 1.12, MidstanceValley: 0.75},
		Sensor:   SensorSpec{Preset: "sm24", TargetDamping: 0.7},
		Geometry: Geometry{Range: 10, ApproachLength: 10},
		Walk:     WalkSpec{Speed: 1.3, Width: 0.12},
		Sampling: Sampling{Rate: 2000, Lead: 0.2, Tail: 1.0},
		Noise:    NoiseSpec{Enabled: true, Seed: 1},
	}
}

// Presets available to SoilSpec.Preset.
var soilPresets = map[string]func() soil.HalfSpace{
	"dry_sand":       soil.DrySand,
	"loam":           soil.Loam,
	"firm_soil":      soil.FirmSoil,
	"weathered_rock": soil.WeatheredRock,
}

// SoilPresetNames lists the presets, for error messages and help text.
func SoilPresetNames() []string {
	names := make([]string, 0, len(soilPresets))
	for n := range soilPresets {
		names = append(names, n)
	}
	return names
}

// Resolved is a Config with every preset expanded and every default made
// explicit. It is what actually gets hashed and what gets written alongside
// the trace, so that a run is reproducible from its own output.
type Resolved struct {
	Soil     soil.HalfSpace    `json:"soil"`
	Walker   grf.Stance        `json:"walker"`
	Sensor   geophone.Geophone `json:"sensor"`
	Geometry Geometry          `json:"geometry"`
	Walk     WalkSpec          `json:"walk"`
	Sampling Sampling          `json:"sampling"`
	Noise    NoiseSpec         `json:"noise"`
}

// Resolve expands presets, applies defaults, and validates every stage.
func (c Config) Resolve() (Resolved, error) {
	var r Resolved

	// Soil: a preset, or explicit constants, but not a contradictory mixture.
	switch {
	case c.Soil.Preset != "":
		make, ok := soilPresets[c.Soil.Preset]
		if !ok {
			return r, fmt.Errorf("config: unknown soil preset %q (have %s)",
				c.Soil.Preset, strings.Join(SoilPresetNames(), ", "))
		}
		r.Soil = make()
		// Explicit fields override the preset, so a config can say "loam but
		// lossier" without restating the whole medium.
		if c.Soil.Vp > 0 {
			r.Soil.Vp = units.SpeedMPS(c.Soil.Vp)
		}
		if c.Soil.Vs > 0 {
			r.Soil.Vs = units.SpeedMPS(c.Soil.Vs)
		}
		if c.Soil.Density > 0 {
			r.Soil.Density = units.DensityKgM3(c.Soil.Density)
		}
		if c.Soil.Qs > 0 {
			r.Soil.Qs = c.Soil.Qs
		}
	default:
		r.Soil = soil.HalfSpace{
			Vp: units.SpeedMPS(c.Soil.Vp), Vs: units.SpeedMPS(c.Soil.Vs),
			Density: units.DensityKgM3(c.Soil.Density), Qs: c.Soil.Qs,
		}
	}
	if err := r.Soil.Validate(); err != nil {
		return r, err
	}

	// Walker.
	r.Walker = grf.Stance{
		Mass:            units.Kilograms(c.Walker.Mass),
		Duration:        units.Seconds(c.Walker.StanceDuration),
		FirstPeak:       c.Walker.FirstPeak,
		SecondPeak:      c.Walker.SecondPeak,
		MidstanceValley: c.Walker.MidstanceValley,
		TransientPeak:   c.Walker.TransientPeak,
		TransientRise:   units.Seconds(c.Walker.TransientRise),
		APPeak:          c.Walker.APPeak,
	}
	if err := r.Walker.Validate(); err != nil {
		return r, err
	}

	// Sensor.
	switch c.Sensor.Preset {
	case "", "sm24":
		r.Sensor = geophone.SM24()
	default:
		return r, fmt.Errorf("config: unknown sensor preset %q (have sm24)", c.Sensor.Preset)
	}
	if c.Sensor.NaturalFreq > 0 {
		r.Sensor.NaturalFreq = units.Hertz(c.Sensor.NaturalFreq)
	}
	if c.Sensor.Sensitivity > 0 {
		r.Sensor.Sensitivity = units.VoltsPerMPS(c.Sensor.Sensitivity)
	}
	if c.Sensor.CoilResistance > 0 {
		r.Sensor.CoilResistance = units.Ohms(c.Sensor.CoilResistance)
	}
	if c.Sensor.MovingMass > 0 {
		r.Sensor.MovingMass = units.Kilograms(c.Sensor.MovingMass)
	}
	if c.Sensor.OpenCircuitDamped > 0 {
		r.Sensor.OpenCircuitDamping = c.Sensor.OpenCircuitDamped
	}
	if c.Sensor.CouplingFreq > 0 {
		r.Sensor.Coupling.ResonanceFreq = units.Hertz(c.Sensor.CouplingFreq)
	}
	if c.Sensor.CouplingDamping > 0 {
		r.Sensor.Coupling.Damping = c.Sensor.CouplingDamping
	}
	if c.Sensor.TargetDamping > 0 {
		rs, err := r.Sensor.ShuntForDamping(c.Sensor.TargetDamping)
		if err != nil {
			return r, err
		}
		r.Sensor.ShuntResistance = rs
	}
	if err := r.Sensor.Validate(); err != nil {
		return r, err
	}

	// Geometry and sampling.
	if c.Geometry.Range <= 0 {
		return r, fmt.Errorf("config: range must be positive, got %g m", c.Geometry.Range)
	}
	if c.Sampling.Rate <= 0 {
		return r, fmt.Errorf("config: sample rate must be positive, got %g Hz", c.Sampling.Rate)
	}
	if c.Sampling.Lead < 0 || c.Sampling.Tail < 0 {
		return r, fmt.Errorf("config: lead and tail must not be negative, got %g and %g s", c.Sampling.Lead, c.Sampling.Tail)
	}
	if c.Geometry.ApproachLength < 0 {
		return r, fmt.Errorf("config: approach length must not be negative, got %g m", c.Geometry.ApproachLength)
	}
	if c.Walk.Speed < 0 {
		return r, fmt.Errorf("config: walking speed must not be negative, got %g m/s", c.Walk.Speed)
	}
	if c.Walk.Width < 0 {
		return r, fmt.Errorf("config: stance width must not be negative, got %g m", c.Walk.Width)
	}
	r.Geometry, r.Walk, r.Sampling, r.Noise = c.Geometry, c.Walk, c.Sampling, c.Noise
	if r.Geometry.ApproachLength == 0 {
		r.Geometry.ApproachLength = 10
	}
	if r.Walk.Speed == 0 {
		r.Walk.Speed = 1.3
	}
	if r.Walk.Width == 0 {
		r.Walk.Width = 0.12
	}
	return r, nil
}

// Hash is the SHA-256 of the resolved configuration, rendered as JSON with
// sorted keys. Two runs share a hash exactly when they describe the same
// physics.
func (r Resolved) Hash() (string, error) {
	// encoding/json already emits struct fields in declaration order and map
	// keys sorted, so the encoding is canonical for this shape.
	b, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// Parse reads a Config from JSON, rejecting unknown fields so that a typo in a
// key is an error rather than a silently ignored setting.
func Parse(data []byte) (Config, error) {
	c := Default()
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("config: %w", err)
	}
	return c, nil
}

// WalkPast builds the walk this configuration describes: the sensor at the
// origin, the walker travelling along +x on a line offset by Range, starting
// ApproachLength before closest approach.
func (r Resolved) WalkPast() source.Walk {
	return source.Walk{
		Stance: r.Walker,
		Speed:  r.Walk.Speed,
		StartX: -r.Geometry.ApproachLength,
		StartY: r.Geometry.Range,
		Width:  r.Walk.Width,
		Until:  r.WalkDuration(),
	}
}

// WalkDuration is how long the walker takes to cross the configured path.
func (r Resolved) WalkDuration() units.Seconds {
	return units.Seconds(2 * r.Geometry.ApproachLength / r.Walk.Speed)
}
