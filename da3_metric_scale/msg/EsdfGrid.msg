# Dense 3D ESDF grid (axis-aligned) from nvblox, in header.frame_id.
#
# Voxel (ix, iy, iz) has its centre at
#   origin + (ix + 0.5, iy + 0.5, iz + 0.5) * voxel_size_m
# and its distance at distances[ix * ny * nz + iy * nz + iz]   (metres; z fastest).
# Positive = free space distance to the nearest surface, negative = inside an obstacle.
# Voxels never observed hold `unobserved_value`.
std_msgs/Header header
geometry_msgs/Point origin        # minimal corner of the minimal voxel
float32 voxel_size_m
uint32 nx
uint32 ny
uint32 nz
float32 unobserved_value
float32[] distances
