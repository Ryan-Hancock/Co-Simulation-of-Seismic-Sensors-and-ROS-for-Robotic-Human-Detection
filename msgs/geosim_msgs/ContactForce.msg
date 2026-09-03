# A force applied to the ground at a point: one foot during stance, or one
# wheel or track segment of a robot.
#
# Deliberately not footstep-specific. O3 predicts the robot's own seismic
# signature "via the same Green's function machinery" as a human footstep, so
# the source stage has to accept wheel and track contacts on equal terms. That
# is a design decision taken now because retrofitting it later would mean
# rewriting every consumer.

std_msgs/Header header

# Which contact this is: "left_foot", "wheel_fr", "track_l_seg3". Stable across
# a run, so a consumer can keep per-contact filter state.
string contact_id

# Contact point in the world frame, metres. Each contact radiates from its own
# position — collapsing several into one point source would throw away the
# spatial structure that makes localisation possible.
float64[3] position

# Force applied to the ground, newtons, world frame. Vertical dominates for
# Rayleigh excitation but the horizontal components are not negligible.
float64[3] force
