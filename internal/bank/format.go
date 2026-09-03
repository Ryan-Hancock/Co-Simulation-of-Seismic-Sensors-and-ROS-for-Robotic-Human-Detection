// Package bank stores precomputed Green's functions and serves them at
// arbitrary range.
//
// This is the file that makes real-time synthesis possible and, separately, the
// file that is the language seam. Computing a wavenumber integral costs
// milliseconds per frequency, which is hopeless inside a chunk; computing it
// once into a bank and interpolating costs microseconds. And because a bank is
// a file rather than a function call, the Python side can write one — from
// Devito or SPECFEM3D — that the Go runtime reads without either process
// knowing the other exists.
//
// # What is stored, and why it is the frequency domain
//
// A bank holds the complex frequency response at each of a grid of ranges,
// not the time-domain impulse response. Two reasons, and the second is the
// important one.
//
// The synthesis path already works in the frequency domain: it multiplies the
// propagation response by the sensor's and inverts once. Storing time-domain
// responses would mean transforming forward again to do that.
//
// More importantly, interpolation between grid ranges is only well behaved in
// the frequency domain. Interpolating two time-domain impulse responses
// linearly does not produce the response at the intermediate range — it
// produces the *average of two arrivals at different times*, which is a
// waveform with two peaks where the real one has one. In the frequency domain
// the phase is very nearly linear in range and the log-amplitude nearly so, and
// interpolating those two quantities separately reproduces a single arrival at
// the right time. That is the difference between a grid that must be finer than
// a fraction of a wavelength and one that can be metres coarse.
//
// # Layout
//
//	magic      8 bytes   "GEOBANK\x00"
//	version    uint32    format version, little-endian
//	hdrLen     uint32    length of the JSON header
//	header     hdrLen    JSON, self-describing
//	padding              zeros, to the next page boundary
//	payload              float32 pairs (re, im), little-endian
//
// The payload is a flat array of Count*(Samples/2+1) complex values in
// range-major order, page-aligned so the whole thing can be mapped rather than
// read when banks grow past comfortable memory. Everything needed to interpret
// it — the medium, the sampling, the range grid, what the numbers physically
// are — is in the header, so a bank found on disk years later is still
// readable.
package bank

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/units"
)

// FormatVersion is the current on-disk format. Versioned from the start so
// WP2 does not break on a mid-project change.
const FormatVersion = 1

var magic = [8]byte{'G', 'E', 'O', 'B', 'A', 'N', 'K', 0}

// pageSize is the payload alignment. Not the true page size of any particular
// machine — a fixed value keeps the format portable, and 4096 divides every
// page size in practice.
const pageSize = 4096

// Provenance records where a bank came from, so a figure in the eventual
// validation report can be traced back to the run that produced it.
type Provenance struct {
	// Solver names what computed the responses: "fk" for this project's
	// wavenumber integration, or the name and version of an external code.
	Solver string `json:"solver"`
	// ConfigHash is the resolved configuration this was built from, if any.
	ConfigHash string `json:"config_hash,omitempty"`
	// Created is an RFC 3339 timestamp.
	Created string `json:"created,omitempty"`
	// Notes is free text for anything the fields above cannot carry.
	Notes string `json:"notes,omitempty"`
}

// RangeGrid is the set of source-receiver distances a bank covers.
//
// Linear rather than logarithmic, deliberately. Phase is very nearly
// proportional to range, so linear spacing makes the phase interpolation
// between neighbours essentially exact — which is where the accuracy of the
// whole scheme comes from.
type RangeGrid struct {
	MinM  float64 `json:"min_m"`
	MaxM  float64 `json:"max_m"`
	Count int     `json:"count"`
}

// At returns the i'th range in the grid.
func (g RangeGrid) At(i int) float64 {
	if g.Count <= 1 {
		return g.MinM
	}
	return g.MinM + (g.MaxM-g.MinM)*float64(i)/float64(g.Count-1)
}

// Spacing is the distance between neighbouring grid ranges.
func (g RangeGrid) Spacing() float64 {
	if g.Count <= 1 {
		return 0
	}
	return (g.MaxM - g.MinM) / float64(g.Count-1)
}

// Header is everything needed to interpret a bank's payload.
type Header struct {
	FormatVersion int         `json:"format_version"`
	Provenance    Provenance  `json:"provenance"`
	Medium        layer.Stack `json:"medium"`
	// SampleRateHz and Samples define the frequency grid: bin k is at
	// k*SampleRateHz/Samples, for k from 0 to Samples/2 inclusive. Expressed
	// this way rather than as an explicit frequency list because it has to
	// match the transform the consumer will invert with, and a list invites
	// the two to drift apart.
	SampleRateHz float64   `json:"sample_rate_hz"`
	Samples      int       `json:"samples"`
	Ranges       RangeGrid `json:"ranges"`
	// Component says what the numbers physically are, in words, because a
	// bank of the wrong component looks entirely plausible.
	Component string `json:"component"`
	Units     string `json:"units"`
}

// Bins is the number of frequency bins per range.
func (h Header) Bins() int { return h.Samples/2 + 1 }

// FrequencyAt is the frequency of bin k.
func (h Header) FrequencyAt(k int) float64 {
	return float64(k) * h.SampleRateHz / float64(h.Samples)
}

// Validate reports whether a header describes a usable bank.
func (h Header) Validate() error {
	switch {
	case h.FormatVersion != FormatVersion:
		return fmt.Errorf("bank: format version %d, this build reads %d", h.FormatVersion, FormatVersion)
	case h.Samples <= 0 || h.Samples%2 != 0:
		return fmt.Errorf("bank: samples must be positive and even, got %d", h.Samples)
	case h.SampleRateHz <= 0:
		return fmt.Errorf("bank: sample rate must be positive, got %g", h.SampleRateHz)
	case h.Ranges.Count <= 0:
		return fmt.Errorf("bank: range grid must have at least one entry, got %d", h.Ranges.Count)
	case h.Ranges.MinM <= 0:
		return fmt.Errorf("bank: minimum range must be positive, got %g m", h.Ranges.MinM)
	case h.Ranges.Count > 1 && h.Ranges.MaxM <= h.Ranges.MinM:
		return fmt.Errorf("bank: maximum range %g m must exceed minimum %g m", h.Ranges.MaxM, h.Ranges.MinM)
	case h.Component == "":
		return fmt.Errorf("bank: component must be named")
	}
	if len(h.Medium) > 0 {
		if err := h.Medium.Validate(); err != nil {
			return fmt.Errorf("bank: %w", err)
		}
	}
	return nil
}

// Bank is a loaded Green's function bank.
type Bank struct {
	Header
	// data holds Count*Bins complex values as interleaved float32 pairs, in
	// range-major order.
	data []float32
	interpTables
}

// New allocates an empty bank matching a header.
func New(h Header) (*Bank, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	return &Bank{Header: h, data: make([]float32, 2*h.Ranges.Count*h.Bins())}, nil
}

// Set stores one response.
func (b *Bank) Set(rangeIndex, bin int, v complex128) error {
	i, err := b.offset(rangeIndex, bin)
	if err != nil {
		return err
	}
	b.data[i] = float32(real(v))
	b.data[i+1] = float32(imag(v))
	return nil
}

// At returns one stored response, without interpolation.
func (b *Bank) At(rangeIndex, bin int) (complex128, error) {
	i, err := b.offset(rangeIndex, bin)
	if err != nil {
		return 0, err
	}
	return complex(float64(b.data[i]), float64(b.data[i+1])), nil
}

func (b *Bank) offset(rangeIndex, bin int) (int, error) {
	if rangeIndex < 0 || rangeIndex >= b.Ranges.Count {
		return 0, fmt.Errorf("bank: range index %d out of range [0, %d)", rangeIndex, b.Ranges.Count)
	}
	if bin < 0 || bin >= b.Bins() {
		return 0, fmt.Errorf("bank: bin %d out of range [0, %d)", bin, b.Bins())
	}
	return 2 * (rangeIndex*b.Bins() + bin), nil
}

// Write serialises a bank to a file.
func Write(path string, b *Bank) error {
	if err := b.Header.Validate(); err != nil {
		return err
	}
	if want := 2 * b.Ranges.Count * b.Bins(); len(b.data) != want {
		return fmt.Errorf("bank: payload has %d floats, header implies %d", len(b.data), want)
	}

	hdr, err := json.Marshal(b.Header)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	buf.Write(magic[:])
	binary.Write(&buf, binary.LittleEndian, uint32(FormatVersion))
	binary.Write(&buf, binary.LittleEndian, uint32(len(hdr)))
	buf.Write(hdr)
	// Pad so the payload starts on a page boundary and the file can be mapped.
	for buf.Len()%pageSize != 0 {
		buf.WriteByte(0)
	}
	if err := binary.Write(&buf, binary.LittleEndian, b.data); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// Open reads a bank from a file.
//
// Reads rather than maps. The layout is page-aligned and flat precisely so a
// mapping implementation can be dropped in when banks outgrow memory, but for
// the sizes this project produces — a laterally homogeneous medium collapses
// to a single range axis, so a bank is megabytes — reading is simpler and has
// no platform-specific code.
func Open(path string) (*Bank, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 16 || !bytes.Equal(raw[:8], magic[:]) {
		return nil, fmt.Errorf("bank: %s is not a Green's function bank", path)
	}
	version := binary.LittleEndian.Uint32(raw[8:12])
	if version != FormatVersion {
		return nil, fmt.Errorf("bank: %s is format version %d, this build reads %d", path, version, FormatVersion)
	}
	hdrLen := int(binary.LittleEndian.Uint32(raw[12:16]))
	if 16+hdrLen > len(raw) {
		return nil, fmt.Errorf("bank: %s has a header length of %d that runs past the file", path, hdrLen)
	}
	var h Header
	if err := json.Unmarshal(raw[16:16+hdrLen], &h); err != nil {
		return nil, fmt.Errorf("bank: %s: parsing header: %w", path, err)
	}
	if err := h.Validate(); err != nil {
		return nil, err
	}

	off := 16 + hdrLen
	if off%pageSize != 0 {
		off += pageSize - off%pageSize
	}
	want := 2 * h.Ranges.Count * h.Bins()
	if (len(raw)-off)/4 < want {
		return nil, fmt.Errorf("bank: %s has %d payload floats, header implies %d", path, (len(raw)-off)/4, want)
	}
	data := make([]float32, want)
	if err := binary.Read(bytes.NewReader(raw[off:]), binary.LittleEndian, data); err != nil {
		return nil, err
	}
	return &Bank{Header: h, data: data}, nil
}

// SizeBytes is how much the payload occupies on disk.
func (b *Bank) SizeBytes() int { return 4 * len(b.data) }

// RangeOf is the range at grid index i.
func (b *Bank) RangeOf(i int) units.Metres { return units.Metres(b.Ranges.At(i)) }

// bracket finds the grid interval containing r, and how far into it r lies.
func (b *Bank) bracket(r float64) (lo int, frac float64, err error) {
	g := b.Ranges
	if r < g.MinM || r > g.MaxM {
		return 0, 0, fmt.Errorf("bank: range %g m is outside the bank's %g to %g m", r, g.MinM, g.MaxM)
	}
	if g.Count == 1 {
		return 0, 0, nil
	}
	x := (r - g.MinM) / g.Spacing()
	lo = int(math.Floor(x))
	if lo >= g.Count-1 {
		return g.Count - 2, 1, nil
	}
	return lo, x - float64(lo), nil
}
