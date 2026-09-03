// Package layer describes a horizontally layered elastic medium: a stack of
// homogeneous layers over a half-space.
//
// This is the medium slice 3 exists for. A real site is layered — soil over
// weathered rock over bedrock — and layering is what makes surface waves
// disperse. Slice 0's homogeneous half-space has a single Rayleigh velocity;
// a layered one has a different velocity at every frequency, because long
// waves sample deep and fast material while short waves stay in the slow
// surface layer. That dispersion is not a correction to the homogeneous
// answer, it is the dominant feature of a real footstep waveform at range.
package layer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// Layer is one homogeneous horizontal layer.
type Layer struct {
	// Thickness in metres. Ignored for the final entry of a Stack, which is
	// the half-space and extends downward without end.
	Thickness units.Metres      `json:"thickness_m"`
	Vp        units.SpeedMPS    `json:"vp_mps"`
	Vs        units.SpeedMPS    `json:"vs_mps"`
	Density   units.DensityKgM3 `json:"density_kgm3"`
	// Qs is the shear quality factor. Zero means lossless, which is what the
	// dispersion solver wants: the elastic dispersion curve is computed first
	// and attenuation applied afterwards, so that the two effects stay
	// separable and each can be tested on its own.
	Qs float64 `json:"qs,omitempty"`
}

// Stack is a layered medium: zero or more layers over a half-space, which is
// always the last entry.
type Stack []Layer

// Uniform is a Stack with no layering, equivalent to a homogeneous half-space.
// It exists so the layered solver can be checked against the closed-form
// answer slice 0 already computes.
func Uniform(h soil.HalfSpace) Stack {
	return Stack{{Vp: h.Vp, Vs: h.Vs, Density: h.Density, Qs: h.Qs}}
}

// HalfSpace is the bottom layer.
func (s Stack) HalfSpace() Layer { return s[len(s)-1] }

// Validate reports whether the stack describes a physically possible medium.
func (s Stack) Validate() error {
	if len(s) == 0 {
		return fmt.Errorf("layer: a stack needs at least a half-space")
	}
	for i, l := range s {
		switch {
		case l.Vs <= 0:
			return fmt.Errorf("layer %d: Vs must be positive, got %g m/s", i, l.Vs)
		case l.Vp <= 0:
			return fmt.Errorf("layer %d: Vp must be positive, got %g m/s", i, l.Vp)
		case l.Density <= 0:
			return fmt.Errorf("layer %d: density must be positive, got %g kg/m^3", i, l.Density)
		case float64(l.Vp) <= math.Sqrt2*float64(l.Vs):
			return fmt.Errorf("layer %d: Vp=%g must exceed sqrt(2)*Vs=%g for a non-negative Poisson ratio",
				i, l.Vp, math.Sqrt2*float64(l.Vs))
		case i < len(s)-1 && l.Thickness <= 0:
			return fmt.Errorf("layer %d: thickness must be positive, got %g m", i, l.Thickness)
		}
	}
	return nil
}

// ShearModulus of a layer.
func (l Layer) ShearModulus() float64 {
	return float64(l.Density) * float64(l.Vs) * float64(l.Vs)
}

// LameLambda of a layer.
func (l Layer) LameLambda() float64 {
	vp, vs := float64(l.Vp), float64(l.Vs)
	return float64(l.Density) * (vp*vp - 2*vs*vs)
}

// VelocityBounds are the slowest and fastest shear velocities in the stack.
//
// They bracket where Rayleigh roots can be: no mode travels slower than the
// Rayleigh velocity of the slowest layer, and none faster than the shear
// velocity of the fastest — above that the wave is not trapped at all. The
// root finder needs the bracket, and getting it wrong is how a mode goes
// missing.
func (s Stack) VelocityBounds() (slowest, fastest units.SpeedMPS) {
	slowest, fastest = s[0].Vs, s[0].Vs
	for _, l := range s {
		slowest = min(slowest, l.Vs)
		fastest = max(fastest, l.Vs)
	}
	return slowest, fastest
}

// TotalThickness is the depth to the top of the half-space.
func (s Stack) TotalThickness() units.Metres {
	var d units.Metres
	for _, l := range s[:len(s)-1] {
		d += l.Thickness
	}
	return d
}

// Golden is a dispersion curve set written by py/oracles/dispersion.py, used
// as the oracle for the Go solver.
//
// It is committed to testdata rather than regenerated, so the Go tests have no
// Python dependency at all — the seam is crossed once, when the file is
// written, and never at test time.
type Golden struct {
	Name   string `json:"name"`
	Why    string `json:"why"`
	Source string `json:"source"`
	Units  string `json:"units"`
	Layers Stack  `json:"layers"`
	Modes  map[string]struct {
		Frequency     []float64 `json:"frequency_hz"`
		PhaseVelocity []float64 `json:"phase_velocity_mps"`
	} `json:"modes"`
}

// LoadGolden reads one golden dispersion file.
func LoadGolden(path string) (Golden, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Golden{}, err
	}
	var g Golden
	if err := json.Unmarshal(data, &g); err != nil {
		return Golden{}, fmt.Errorf("layer: parsing %s: %w", path, err)
	}
	if err := g.Layers.Validate(); err != nil {
		return Golden{}, fmt.Errorf("layer: %s: %w", path, err)
	}
	return g, nil
}
