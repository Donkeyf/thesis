#!/usr/bin/env python3
"""Convert a bag of da3_metric_scale/msg/EsdfGrid into standard sensor_msgs/PointCloud2
(viewable in any RViz / Foxglove without this package) and optionally one merged
whole-map PLY.

Run where da3_metric_scale is built and sourced (the Jetson container):
  source /workspaces/isaac_ros-dev/install/setup.bash
  python3 esdf_bag_to_cloud.py esdf_run1 -o esdf_run1_cloud --ply esdf_run1_map.ply

  input   bag directory or a single .mcap file (recover broken MCAPs first:
          `mcap recover in.mcap -o fixed.mcap`)
  -o      output bag directory (MCAP) with /esdf_grid_cloud (x, y, z, distance)
  --ply   merged map: every voxel ever observed, newest value wins (CloudCompare:
          colour by scalar field 'distance')
  --max-distance  keep voxels with ESDF <= this (default 0.10 m = obstacle surfaces;
                  -1 = all observed voxels)

View on any machine:
  ros2 bag play esdf_run1_cloud --loop
  rviz2   -> Fixed Frame map, PointCloud2 /esdf_grid_cloud, Intensity channel 'distance',
             Decay Time 1000 (accumulates the whole map)
"""
import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------- pure numpy
def grid_points(g, max_d, min_d):
    """EsdfGrid -> (N,4) float32 [x, y, z, distance] and (N,3) int64 global voxel idx."""
    nx, ny, nz = int(g.nx), int(g.ny), int(g.nz)
    d = np.asarray(g.distances, dtype=np.float32)
    if d.size != nx * ny * nz or d.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, 3), np.int64)
    d = d.reshape(nx, ny, nz)                     # index (ix*ny + iy)*nz + iz
    keep = np.abs(d - g.unobserved_value) > 1e-2
    if max_d > -1:
        keep &= (d <= max_d) & (d >= min_d)
    ix, iy, iz = np.nonzero(keep)
    vs = float(g.voxel_size_m)
    o = np.array([g.origin.x, g.origin.y, g.origin.z])
    idx = np.stack([ix, iy, iz], 1).astype(np.int64)
    centres = o + (idx + 0.5) * vs
    gidx = np.rint(centres / vs - 0.5).astype(np.int64)   # global voxel index
    pts = np.empty((ix.size, 4), np.float32)
    pts[:, :3] = centres
    pts[:, 3] = d[ix, iy, iz]
    return pts, gidx


class VoxelMerge:
    """Whole-map accumulator: newest value per global voxel index."""

    def __init__(self):
        self.idx = np.zeros((0, 3), np.int64)
        self.val = np.zeros((0, 4), np.float32)

    def add(self, gidx, pts):
        if gidx.shape[0] == 0:
            return
        idx = np.concatenate([gidx, self.idx])          # new first -> wins in unique()
        val = np.concatenate([pts, self.val])
        _, first = np.unique(idx, axis=0, return_index=True)
        self.idx, self.val = idx[first], val[first]


def write_ply(path, pts):
    with open(path, 'wb') as f:
        f.write((
            'ply\nformat binary_little_endian 1.0\n'
            f'element vertex {pts.shape[0]}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property float distance\nend_header\n').encode())
        f.write(np.ascontiguousarray(pts, dtype='<f4').tobytes())


# ---------------------------------------------------------------- ROS bag I/O
def make_cloud(header, pts):
    from sensor_msgs.msg import PointCloud2, PointField
    pc = PointCloud2()
    pc.header = header
    pc.height = 1
    pc.width = int(pts.shape[0])
    pc.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                 for i, n in enumerate(('x', 'y', 'z', 'distance'))]
    pc.is_bigendian = False
    pc.point_step = 16
    pc.row_step = 16 * pc.width
    pc.is_dense = True
    pc.data = pts.astype('<f4').tobytes()
    return pc


def topic_metadata(name, type_):
    import rosbag2_py
    try:                                   # Jazzy+ (has an id field)
        return rosbag2_py.TopicMetadata(id=0, name=name, type=type_, serialization_format='cdr')
    except TypeError:                      # Humble
        return rosbag2_py.TopicMetadata(name=name, type=type_, serialization_format='cdr')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('-o', '--out', default=None, help='output bag dir (default <bag>_cloud)')
    ap.add_argument('--topic', default='/esdf_publisher/esdf_grid')
    ap.add_argument('--out-topic', default='/esdf_grid_cloud')
    ap.add_argument('--max-distance', type=float, default=0.10)
    ap.add_argument('--min-distance', type=float, default=-1.0)
    ap.add_argument('--every', type=int, default=1, help='use every Nth grid')
    ap.add_argument('--ply', default=None, help='also write a merged whole-map PLY')
    ap.add_argument('--no-bag', action='store_true', help='only write the PLY')
    args = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    try:
        from da3_metric_scale.msg import EsdfGrid
    except ImportError:
        sys.exit('cannot import da3_metric_scale.msg - source the workspace where '
                 'da3_metric_scale is built')

    src = args.bag.rstrip('/')
    storage = 'mcap' if src.endswith('.mcap') else ''
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(rosbag2_py.StorageOptions(uri=src, storage_id=storage),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
    except Exception as e:  # noqa: BLE001
        sys.exit(f'cannot open {src}: {e}\n(an MCAP that was not closed cleanly needs '
                 '`mcap recover in.mcap -o fixed.mcap` first)')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    writer = None
    if not args.no_bag:
        out = args.out or (os.path.splitext(src)[0] + '_cloud')
        writer = rosbag2_py.SequentialWriter()
        writer.open(rosbag2_py.StorageOptions(uri=out, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        writer.create_topic(topic_metadata(args.out_topic, 'sensor_msgs/msg/PointCloud2'))

    merge = VoxelMerge() if args.ply else None
    n_in = n_out = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != args.topic:
            continue
        n_in += 1
        if (n_in - 1) % args.every:
            continue
        g = deserialize_message(data, EsdfGrid)
        pts, gidx = grid_points(g, args.max_distance, args.min_distance)
        if writer is not None:
            writer.write(args.out_topic, serialize_message(make_cloud(g.header, pts)), t)
        if merge is not None:
            merge.add(gidx, pts)
        n_out += 1
        if n_out % 50 == 0:
            print(f'  {n_out} grids converted'
                  + (f', merged map {merge.val.shape[0]} voxels' if merge else ''))

    if writer is not None:
        del writer                          # closes the bag and writes the MCAP index
        print(f'wrote {n_out} PointCloud2 messages to {out} ({args.out_topic})')
    if merge is not None:
        write_ply(args.ply, merge.val)
        print(f'wrote merged map: {merge.val.shape[0]} voxels -> {args.ply}')
    if n_in == 0:
        print(f'no {args.topic} messages found')


if __name__ == '__main__':
    main()
