# One contiguous block of geophone samples.
#
# Published as a chunk rather than a sample at a time. A geophone runs at
# 1-2 kHz, and one message per sample would be a couple of thousand messages a
# second per sensor through a node executor, for no benefit: a 10 ms chunk at
# 2 kHz is 20 samples and 100 messages a second.
#
# The chunk length is not a constant. It sets the coupling granularity between
# Isaac, which ticks at 60-240 Hz, and the seismic stream, so it is the knob
# O2 sweeps when it characterises extrapolation error at the interface.

# header.stamp is the simulation time of samples[0], not of publication.
# Everything downstream reconstructs sample times from it and sample_rate_hz,
# so it has to be the former.
std_msgs/Header header

# Sampling rate in hertz. Carried per chunk rather than assumed, so a recording
# stays interpretable on its own.
float64 sample_rate_hz

# Geophone output in volts, oldest first. float32 is deliberate: the sensor's
# own noise floor is far above float32 resolution at these amplitudes, so the
# extra precision would only cost bandwidth.
float32[] samples

# Source-to-receiver range used to synthesise this chunk, in metres. Present so
# a recorded stream can be checked against the geometry that produced it.
float64 range_m
