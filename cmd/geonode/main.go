// Command geonode publishes a synthetic geophone stream onto a ROS 2 graph.
//
// This is slice 1 of the WP1 plan: the forward model, running continuously
// under simulated time, emitting chunks a real ROS node can subscribe to. The
// physics is still slice 0's — one foot, a homogeneous half-space, far field —
// and deliberately so. What is being proven here is the transport and the
// timing, before any of the harder physics comes to depend on them.
//
//	geonode                                    # in-process, wall clock
//	geonode -use-sim-time                      # driven by /clock
//	geonode -transport zenoh -use-sim-time     # on a live ROS 2 graph
//
// Under simulated time nothing is published until the simulator says what time
// it is — Conductor gates timers on the clock having started — which is the
// behaviour that makes a co-simulation reproducible: the seismic stream
// advances with Isaac rather than alongside it.
package main

import (
	"log/slog"
	"os"
	"strings"
	"time"

	"conductor.dev/conductor"
	_ "conductor.dev/conductor/transport/zenoh"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/sensing"
	"geosim.dev/geosim/msgs"
)

// chunkRateHz is how often a chunk is published. It has to agree with the rate
// tag on the timer below, and OnTick checks that it does — the tag is a
// compile-time string while the chunk size is computed from this constant, so
// nothing else would catch them drifting apart.
//
// 100 Hz means a 10 ms chunk at 2 kHz: twenty samples, a hundred messages a
// second. One message per sample would be two thousand, for no gain. The value
// is not sacred — it is the coupling granularity between Isaac's 60-240 Hz tick
// and the seismic stream, and so the knob O2 sweeps to characterise interface
// error.
const chunkRateHz = 100

//conductor:node
type Geophone struct {
	// Tick drives synthesis. Under -use-sim-time it follows /clock, so the
	// stream advances with the simulator rather than with the wall.
	Tick conductor.Timer `rate:"100hz"`

	// Chunk carries the samples. Sensor QoS, like any high-rate stream: a late
	// chunk is worth less than the next one, and stalling synthesis to
	// retransmit it would be the wrong trade.
	Chunk conductor.Pub[msgs.GeophoneChunk] `topic:"geophone/chunk" qos:"sensor" frame:"geophone_link"`

	engine  *sensing.Engine
	rangeM  float64
	samples []float32 // sized once; each chunk gets its own copy, see OnTick
	checked bool
}

func (g *Geophone) OnTick() {
	if !g.checked {
		g.checked = true
		if want := time.Second / chunkRateHz; g.Tick.Period() != want {
			conductor.Abort(mismatch{g.Tick.Period(), want})
			return
		}
	}
	// A fresh slice per chunk. The engine would happily refill one buffer
	// forever, and over a serialising transport that would be fine — but the
	// in-process transport hands the message across by reference, so a reused
	// buffer means every subscriber holds the same array and sees only the
	// most recent chunk. The bug is invisible in the waveform: each chunk is
	// individually correct, and only the assembled stream is wrong.
	//
	// This is the one allocation in the loop, and it is an ownership transfer
	// rather than waste: twenty float32 a hundred times a second.
	samples := make([]float32, len(g.samples))
	if err := g.engine.Next(samples); err != nil {
		slog.Error("synthesis failed", "err", err)
		return
	}
	// Header.Stamp is left zero so the runtime fills it from the robot's clock
	// — simulated time under -use-sim-time. The timer fires when that clock
	// reaches this chunk's boundary, so the stamp is the time of samples[0],
	// which is what the message contract promises.
	//
	// Sample times within the chunk come from sample_rate_hz, never from when
	// this goroutine happened to wake. How far those two can drift apart under
	// a coarse /clock is an O2 question, and a real one.
	g.Chunk.Publish(msgs.GeophoneChunk{
		SampleRateHz: g.engine.SampleRate(),
		Samples:      samples,
		RangeM:       g.rangeM,
	})
}

type mismatch struct{ got, want time.Duration }

func (m mismatch) Error() string {
	return "geonode: timer period " + m.got.String() + " does not match the chunk rate " + m.want.String()
}

func main() {
	configPath, rest := extractConfigFlag(os.Args[1:])
	os.Args = append(os.Args[:1], rest...)

	cfg := config.Default()
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			slog.Error("geonode: reading config", "err", err)
			os.Exit(1)
		}
		if cfg, err = config.Parse(data); err != nil {
			slog.Error("geonode: parsing config", "err", err)
			os.Exit(1)
		}
	}
	res, err := cfg.Resolve()
	if err != nil {
		slog.Error("geonode: invalid config", "err", err)
		os.Exit(1)
	}

	walk := res.WalkPast()
	if err := walk.Validate(); err != nil {
		slog.Error("geonode: invalid walk", "err", err)
		os.Exit(1)
	}
	chunk := int(res.Sampling.Rate / chunkRateHz)
	engine, err := sensing.NewEngine(res, walk, chunk)
	if err != nil {
		slog.Error("geonode: building the sensing engine", "err", err)
		os.Exit(1)
	}
	hash, _ := res.Hash()
	slog.Info("geonode ready",
		"config", hash[:16],
		"soil", res.Soil.String(),
		"range_m", res.Geometry.Range,
		"sample_rate_hz", res.Sampling.Rate,
		"chunk_samples", chunk,
		"chunk_ms", float64(chunk)/res.Sampling.Rate*1000,
		"closest_approach_m", res.Geometry.Range,
		"walk_speed_mps", res.Walk.Speed,
		"step_period_s", walk.StepPeriod(),
		"walk_duration_s", res.WalkDuration())

	conductor.Run(&Geophone{
		engine:  engine,
		rangeM:  res.Geometry.Range,
		samples: make([]float32, chunk),
	})
}

// extractConfigFlag pulls this command's own -geosim-config out of the
// argument list and returns the rest for conductor.Run to parse. Conductor
// owns the flag set, so the alternative would be teaching it about flags that
// are none of its business.
func extractConfigFlag(args []string) (string, []string) {
	const name = "-geosim-config"
	var path string
	var rest []string
	for i := 0; i < len(args); i++ {
		switch {
		case args[i] == name && i+1 < len(args):
			path = args[i+1]
			i++
		case strings.HasPrefix(args[i], name+"="):
			path = strings.TrimPrefix(args[i], name+"=")
		default:
			rest = append(rest, args[i])
		}
	}
	return path, rest
}
