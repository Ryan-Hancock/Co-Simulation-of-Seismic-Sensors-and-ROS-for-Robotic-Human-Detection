# Co-Simulation of Seismic Sensors and ROS for Robotic Human Detection

### Objectives

**O1 — Physically grounded geophone sensor model.** Develop a forward model that maps a human footstep (ground reaction force, position, time) to the voltage output of a vertical geophone at arbitrary range over a parameterised soil, including Rayleigh-wave geometric spreading, frequency-dependent material attenuation, dispersion in layered media, and the sensor's own transfer function (coil resonance, damping, noise floor).

**O2 — A multi-rate co-simulation framework.** Couple Isaac Sim (robot dynamics, terrain, human motion, optical sensing) to a seismic wave solver via ROS 2, and characterise the coupling itself: step-size selection, extrapolation error at the interface, and achievable real-time factor. Isaac ticks at 60–240 Hz; a geophone samples at ~1 kHz and footstep energy runs to ~100 Hz. Reconciling those rates without corrupting the signal is a research question, not an implementation detail.

**O3 — Detection and localisation from a moving platform.** Existing seismic footstep work assumes a static, buried, surveyed sensor array. A robot carrying geophones is mobile, has unknown and varying ground coupling, and is itself a strong vibration source. Develop ego-noise-aware detection that exploits the robot's own proprioception (wheel/track contact forces, joint states) to predict and subtract its self-signature.

**O4 — Quantified sim-to-real transfer.** Train detection/classification in simulation, validate against field geophone recordings, and quantify the gap. Establish which domain randomisation axes (soil velocity and Q, layer depth, coupling stiffness, footwear, body mass, gait speed) actually close it.

**O5 — Energy-aware sensing policy (the net-zero argument).** A geophone is a passive transducer drawing microwatts; a lidar and its compute draw tens of watts continuously. Demonstrate and quantify a duty-cycling policy where persistent seismic sensing wakes the high-power perception stack only on a candidate detection, and characterise the energy-saving vs detection-latency trade-off.

### Methodology

**WP1 — Forward seismic model and offline Green's functions.**  
The medium is linear and time-invariant, so precompute rather than solve online. Use a 3D elastic finite-difference or spectral-element solver (Devito is a good fit — Python, generates optimised GPU code; SPECFEM3D if you need higher fidelity) to compute Green's functions from a grid of surface source positions to a grid of receiver positions over the terrain. At runtime, a footstep becomes a convolution of the source-time function with an interpolated Green's function. This is what makes real-time co-simulation feasible at all.

The source-time function is the weak link and worth being explicit about: Isaac's animated characters are kinematic, so their contact forces aren't physical. You'll need a parametric ground reaction force model (double-hump vertical GRF, heel-strike transient of a few tens of milliseconds, peak ~1.1–1.3× body weight) driven by gait phase, mass, speed and surface, validated against force-plate data. State this as a modelling assumption with a sensitivity analysis rather than hiding it.

**WP2 — Co-simulation architecture.**  
Isaac Sim publishes footstep events, character pose, robot pose and simulation time over ROS 2. A seismic node consumes these, synthesises waveforms at 1–2 kHz, and publishes as a sensor stream. A custom OmniGraph node handles the Isaac side; `/clock` and sim time carry synchronisation.

Frame this against the co-simulation literature properly — Gauss-Seidel vs Jacobi coupling, zero-order hold extrapolation error, the FMI master-algorithm work. Position it against Gomes et al.'s co-simulation survey so it reads as a contribution to co-simulation methodology, not just an integration exercise. Validate the coupling against a monolithic reference run at a single small step size and report waveform error as a function of coupling step.

**WP3 — Detection, localisation, ego-noise.**  
Baselines first: STA/LTA picking, kurtosis-based impulse detection, matched filtering, TDOA plus hyperbolic localisation across a small onboard array. Then the learned approach, trained on simulated data. Ego-noise handling is the differentiator — predict the robot's own seismic contribution from its commanded and measured contact forces via the same Green's function machinery, and subtract.

**WP4 — Field validation.**  
SM-24 or similar vertical geophones, a small array, a UGV. Collect walk-past data across a few soil types and ranges. Compare simulated and measured waveforms on interpretable statistics — spectral centroid, SNR vs range decay, inter-step interval — rather than sample-wise error, which will never match. Report detection ROC and localisation error for sim-trained models on real data against real-trained models.

**WP5 — Energy and integration.**  
Instrument the power draw of both sensing modes on real hardware, implement the trigger policy, and evaluate on a realistic scenario — a wind farm or solar site inspection route where the robot must detect an unexpected person.