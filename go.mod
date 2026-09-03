module geosim.dev/geosim

go 1.25.0

require gonum.org/v1/gonum v0.17.0

require conductor.dev/conductor v0.0.0

// Conductor is the Go ROS 2 framework this project's WP2 node runs on. It is
// not published, so it is used from the working tree next door.
replace conductor.dev/conductor => ../ros2-framework
