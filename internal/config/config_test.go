package config

import (
	"encoding/json"
	"strings"
	"testing"
)

// The hash identifies the physics, not the spelling. Two configs that resolve
// to the same medium must agree even if one names a preset and the other
// spells it out, and any change to a physical parameter must move it — that is
// what makes it usable as provenance for O4's randomised runs.
func TestHashTracksPhysicsNotSpelling(t *testing.T) {
	byPreset := Default()
	explicit := Default()
	explicit.Soil = SoilSpec{Vp: 500, Vs: 200, Density: 1700, Qs: 25}

	a, err := mustHash(byPreset)
	if err != nil {
		t.Fatal(err)
	}
	b, err := mustHash(explicit)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Errorf("preset and explicit spellings of the same soil hashed differently:\n  %s\n  %s", a, b)
	}

	changed := Default()
	changed.Soil.Qs = 26
	c, err := mustHash(changed)
	if err != nil {
		t.Fatal(err)
	}
	if c == a {
		t.Error("changing Qs did not change the hash")
	}
}

func TestHashIsStable(t *testing.T) {
	a, _ := mustHash(Default())
	for range 5 {
		b, _ := mustHash(Default())
		if a != b {
			t.Fatalf("hash is not deterministic: %s then %s", a, b)
		}
	}
}

func mustHash(c Config) (string, error) {
	r, err := c.Resolve()
	if err != nil {
		return "", err
	}
	return r.Hash()
}

// A typo in a key must be an error, not a silently ignored setting. Getting
// this wrong is how a sensitivity sweep quietly runs the same configuration
// twenty times.
func TestParseRejectsUnknownFields(t *testing.T) {
	_, err := Parse([]byte(`{"soil": {"presett": "loam"}}`))
	if err == nil {
		t.Fatal("Parse accepted a misspelled key")
	}
	if !strings.Contains(err.Error(), "presett") {
		t.Errorf("error should name the offending field, got: %v", err)
	}
}

func TestParseFillsDefaults(t *testing.T) {
	c, err := Parse([]byte(`{"geometry": {"range_m": 25}}`))
	if err != nil {
		t.Fatal(err)
	}
	if c.Geometry.Range != 25 {
		t.Errorf("range = %g, want 25", c.Geometry.Range)
	}
	if c.Sampling.Rate != Default().Sampling.Rate {
		t.Errorf("sample rate = %g, want the default %g", c.Sampling.Rate, Default().Sampling.Rate)
	}
}

// Explicit fields refine a preset rather than replacing it, so a config can
// say "loam but lossier" without restating the medium.
func TestExplicitFieldsOverridePreset(t *testing.T) {
	c := Default()
	c.Soil = SoilSpec{Preset: "loam", Qs: 8}
	r, err := c.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if r.Soil.Qs != 8 {
		t.Errorf("Qs = %g, want the override 8", r.Soil.Qs)
	}
	if r.Soil.Vs != 200 {
		t.Errorf("Vs = %g, want loam's 200", r.Soil.Vs)
	}
}

// The target damping is expressed as a damping ratio, not a resistance,
// because that is the term the response is actually shaped by. Resolve has to
// turn it into the shunt that achieves it.
func TestTargetDampingSelectsShunt(t *testing.T) {
	r, err := Default().Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if got := r.Sensor.Damping(); got < 0.699 || got > 0.701 {
		t.Errorf("resolved damping %g, want 0.7", got)
	}
	if r.Sensor.ShuntResistance <= 0 {
		t.Error("no shunt resistance was selected")
	}
}

func TestResolveRejectsBadConfigs(t *testing.T) {
	for name, mutate := range map[string]func(*Config){
		"unknown soil preset":   func(c *Config) { c.Soil = SoilSpec{Preset: "cheese"} },
		"unknown sensor preset": func(c *Config) { c.Sensor.Preset = "geode" },
		"zero range":            func(c *Config) { c.Geometry.Range = 0 },
		"zero sample rate":      func(c *Config) { c.Sampling.Rate = 0 },
		"negative lead":         func(c *Config) { c.Sampling.Lead = -1 },
		"unphysical soil":       func(c *Config) { c.Soil = SoilSpec{Vp: 100, Vs: 200, Density: 1700, Qs: 20} },
		"zero mass":             func(c *Config) { c.Walker.Mass = 0 },
		"unreachable damping":   func(c *Config) { c.Sensor.TargetDamping = 0.1 },
	} {
		t.Run(name, func(t *testing.T) {
			c := Default()
			mutate(&c)
			if _, err := c.Resolve(); err == nil {
				t.Errorf("Resolve accepted %s", name)
			}
		})
	}
}

// A run must be reproducible from its own output, so the resolved form has to
// survive a round trip through the sidecar.
func TestResolvedRoundTripsThroughJSON(t *testing.T) {
	r, err := Default().Resolve()
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var back Resolved
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	want, _ := r.Hash()
	got, _ := back.Hash()
	if got != want {
		t.Errorf("hash after round trip is %s, want %s", got, want)
	}
}
