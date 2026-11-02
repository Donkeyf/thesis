#!/usr/bin/env python3
"""Plot the map, the flown path and the planned trajectories from a planner rosbag.

Works WITHOUT ROS (any PC, also Windows/macOS): reads mcap or sqlite3 rosbag2 bags with the
pure-Python `rosbags` library.

    pip install rosbags mcap matplotlib numpy
    python3 plot_planner_bag.py planner_run_171530                  # -> planner_run_171530_overview.png
    python3 plot_planner_bag.py planner_run_171530 --pdf --show
    python3 plot_planner_bag.py planner_run_171530 --t0 5 --t1 40   # time window [s from bag start]
    python3 plot_planner_bag.py planner_run_171530 --snapshots 6    # map + active plan at 6 times
    python3 plot_planner_bag.py planner_run_171530 --3d

Topics used (whatever is in the bag; planner prefix fast_planner or raptor_planner is detected):
    map        /esdf_grid_cloud (preferred) | /esdf_publisher/esdf_cloud | /esdf_publisher/esdf_grid
    position   /mavros/mavros/local_position/pose (EKF)   /vrpn_mocap/dilon/pose_throttled (OptiTrack)
    planning   /<planner>/bspline_path  /<planner>/astar_path  /<planner>/current_goal
               /<planner>/yaw_plan (raptor)               /trajectory_tracker/reference
    status     /<planner>/status  /trajectory_tracker/status

Map: every cloud is cut to obstacle surfaces (ESDF <= --surface, above --floor-cut), voxelised,
and a voxel is drawn only if it appeared in >= --min-hits clouds (removes transient monocular
noise). Top view colours voxels by height.
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
except ImportError:
    sys.exit('needs rosbags:  pip install rosbags mcap matplotlib numpy')

ESDF_GRID_MSG = """
std_msgs/Header header
geometry_msgs/Point origin
float32 voxel_size_m
uint32 nx
uint32 ny
uint32 nz
float32 unobserved_value
float32[] distances
"""

MAP_TOPICS = ['/esdf_grid_cloud', '/esdf_publisher/esdf_cloud', '/esdf_publisher/esdf_grid']
EKF_TOPIC = '/mavros/mavros/local_position/pose'
MOCAP_TOPICS = ['/vrpn_mocap/dilon/pose_throttled', '/vrpn_mocap/dilon/pose']
REF_TOPIC = '/trajectory_tracker/reference'
TRK_STATUS = '/trajectory_tracker/status'
PLANNERS = ['fast_planner', 'raptor_planner']


# ------------------------------------------------------------------------------------- reading
def stamp(msg_header, t_log):
    s = msg_header.stamp.sec + msg_header.stamp.nanosec * 1e-9
    return s if s > 0 else t_log


def cloud_xyzd(msg):
    """x, y, z and (if present) distance channel of a PointCloud2 as float arrays."""
    fields = {f.name: f for f in msg.fields}
    n = msg.width * max(1, msg.height)
    if n == 0 or not all(k in fields for k in 'xyz'):
        return np.zeros((0, 4))
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)[: n * msg.point_step].reshape(n, msg.point_step)

    def col(name):
        f = fields[name]
        return raw[:, f.offset:f.offset + 4].copy().view('<f4').ravel().astype(float)

    d = col('distance') if 'distance' in fields else np.zeros(n)
    out = np.column_stack([col('x'), col('y'), col('z'), d])
    return out[np.isfinite(out).all(axis=1)]


def grid_surface(msg, surface):
    """Observed voxel centres of an EsdfGrid with distance <= surface (obstacle surfaces)."""
    d = np.asarray(msg.distances, dtype=np.float32).reshape(msg.nx, msg.ny, msg.nz)
    obs = np.abs(d - msg.unobserved_value) > 1e-2
    idx = np.argwhere(obs & (d <= surface))
    vs = msg.voxel_size_m
    o = np.array([msg.origin.x, msg.origin.y, msg.origin.z])
    pts = o + (idx + 0.5) * vs
    return np.column_stack([pts, d[obs & (d <= surface)]])


class _RosbagsSource:
    """rosbag2 directory (mcap or sqlite3) through rosbags.AnyReader."""

    def __init__(self, path, ts):
        self.reader = AnyReader([Path(path)], default_typestore=ts)
        self.reader.open()
        self.topics = {c.topic: c.msgtype for c in self.reader.connections}
        self.start_ns = self.reader.start_time

    def messages(self, topics):
        conns = [c for c in self.reader.connections if c.topic in topics]
        for conn, t_ns, raw in self.reader.messages(connections=conns):
            yield conn.topic, conn.msgtype, t_ns, raw, self.reader.deserialize

    def close(self):
        self.reader.close()


class _McapSource:
    """.mcap file(s) read directly (no metadata.yaml needed, any rosbag2 metadata version)."""

    def __init__(self, path, ts):
        from mcap.reader import make_reader
        p = Path(path)
        self.files = sorted(p.glob('*.mcap')) if p.is_dir() else [p]
        if not self.files:
            raise FileNotFoundError(f'no .mcap file in {path}')
        self.ts = ts
        self._make_reader = make_reader
        self.topics, self.start_ns = {}, None
        for f in self.files:
            with open(f, 'rb') as fh:
                r = make_reader(fh)
                summ = r.get_summary()
                for ch in summ.channels.values():
                    self.topics[ch.topic] = summ.schemas[ch.schema_id].name
                st = summ.statistics.message_start_time if summ.statistics else None
                if st is not None:
                    self.start_ns = st if self.start_ns is None else min(self.start_ns, st)

    def _register(self, schema):
        """types the typestore does not know: use the definition stored in the mcap schema."""
        if schema.name in self.ts.types or schema.encoding != 'ros2msg':
            return
        text = schema.data.decode()
        defs = {}
        name, buf = schema.name, []
        for line in text.splitlines():
            if line.startswith('====='):
                defs.update(get_types_from_msg('\n'.join(buf), name))
                name, buf = None, []
            elif line.startswith('MSG: ') and name is None:
                name = line[5:].strip()
                if '/msg/' not in name:
                    name = name.replace('/', '/msg/', 1)
            else:
                buf.append(line)
        if name:
            defs.update(get_types_from_msg('\n'.join(buf), name))
        try:
            self.ts.register(defs)
        except Exception as e:  # conflicting nested definition: keep the typestore's own
            print(f'  could not register {schema.name} from the bag: {e}')

    def messages(self, topics):
        deser = self.ts.deserialize_cdr
        for f in self.files:
            with open(f, 'rb') as fh:
                for schema, ch, m in self._make_reader(fh).iter_messages(topics=list(topics),
                                                                         log_time_order=True):
                    if self.start_ns is None:
                        self.start_ns = m.log_time
                    self._register(schema)
                    yield ch.topic, schema.name, m.log_time, m.data, deser

    def close(self):
        pass


def open_bag(path, ts, force_mcap=False):
    if not force_mcap:
        try:
            return _RosbagsSource(path, ts)
        except Exception as e:  # e.g. rosbag2 metadata version newer than rosbags supports
            print(f'  rosbags could not open the bag ({e}); reading the .mcap directly')
    return _McapSource(path, ts)


def read_bag(path, t0=None, t1=None, surface=0.10, force_mcap=False):
    ts = get_typestore(Stores.ROS2_JAZZY)
    ts.register(get_types_from_msg(ESDF_GRID_MSG, 'da3_metric_scale/msg/EsdfGrid'))
    data = {'clouds': [], 'ekf': [], 'mocap': [], 'ref': [], 'plans': [], 'astar': [], 'goals': [],
            'yaw': [], 'pstatus': [], 'tstatus': [], 'map_topic': None, 'planner': None,
            'mocap_frame': None}
    src = open_bag(path, ts, force_mcap)
    try:
        topics = src.topics
        planner = next((p for p in PLANNERS if f'/{p}/bspline_path' in topics or f'/{p}/status' in topics),
                       None)
        data['planner'] = planner
        map_topic = next((t for t in MAP_TOPICS if t in topics), None)
        data['map_topic'] = map_topic
        mocap_topic = next((t for t in MOCAP_TOPICS if t in topics), None)
        want = {map_topic, EKF_TOPIC, mocap_topic, REF_TOPIC, TRK_STATUS}
        if planner:
            want |= {f'/{planner}/{x}' for x in
                     ('bspline_path', 'astar_path', 'current_goal', 'yaw_plan', 'status')}
        want = {t for t in want if t and t in topics}
        print(f'bag: {path}\n  planner: {planner}   map: {map_topic}   mocap: {mocap_topic}')
        print('  topics: ' + ', '.join(sorted(want)))
        missing = [t for t in (EKF_TOPIC, REF_TOPIC) if t not in topics]
        if missing:
            print('  not in bag: ' + ', '.join(missing))
        bad = set()
        for top, msgtype, t_ns, raw, deser in src.messages(want):
            if top in bad:
                continue
            tr = (t_ns - src.start_ns) * 1e-9
            if (t0 is not None and tr < t0) or (t1 is not None and tr > t1):
                continue
            try:
                msg = deser(raw, msgtype)
            except Exception as e:  # unknown type etc.
                print(f'  skip {top}: {e}')
                bad.add(top)
                continue
            if top == map_topic:
                pts = grid_surface(msg, surface) if top.endswith('esdf_grid') else cloud_xyzd(msg)
                data['clouds'].append((tr, pts))
            elif top in (EKF_TOPIC, mocap_topic, REF_TOPIC):
                p, q = msg.pose.position, msg.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                key = {EKF_TOPIC: 'ekf', mocap_topic: 'mocap', REF_TOPIC: 'ref'}[top]
                data[key].append((tr, p.x, p.y, p.z, yaw))
                if key == 'mocap':
                    data['mocap_frame'] = msg.header.frame_id
            elif planner and top == f'/{planner}/bspline_path':
                data['plans'].append((tr, np.array([[s.pose.position.x, s.pose.position.y,
                                                     s.pose.position.z] for s in msg.poses])))
            elif planner and top == f'/{planner}/astar_path':
                data['astar'].append((tr, np.array([[s.pose.position.x, s.pose.position.y,
                                                     s.pose.position.z] for s in msg.poses])))
            elif planner and top == f'/{planner}/current_goal':
                p = msg.pose.position
                data['goals'].append((tr, p.x, p.y, p.z))
            elif planner and top == f'/{planner}/yaw_plan':
                arr = []
                for s in msg.poses:
                    q = s.pose.orientation
                    arr.append([s.pose.position.x, s.pose.position.y, s.pose.position.z,
                                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))])
                data['yaw'].append((tr, np.array(arr)))
            elif planner and top == f'/{planner}/status':
                data['pstatus'].append((tr, msg.data))
            elif top == TRK_STATUS:
                data['tstatus'].append((tr, msg.data))
    finally:
        src.close()
    for k in ('ekf', 'mocap', 'ref'):
        data[k] = np.array(data[k]) if data[k] else np.zeros((0, 5))
    return data


# ------------------------------------------------------------------------------------- map
def accumulate_map(clouds, voxel=0.05, floor_cut=0.15, surface=0.10, min_hits=3, t_max=None):
    """Voxels seen as obstacle surface in >= min_hits clouds (up to t_max)."""
    hits = defaultdict(int)
    for t, pts in clouds:
        if t_max is not None and t > t_max:
            break
        if len(pts) == 0:
            continue
        sel = pts[(pts[:, 2] > floor_cut) & (pts[:, 3] <= surface)]
        if len(sel) == 0:
            continue
        keys = np.unique(np.floor(sel[:, :3] / voxel).astype(np.int64), axis=0)
        for k in map(tuple, keys):
            hits[k] += 1
    keys = np.array([k for k, n in hits.items() if n >= min_hits])
    if len(keys) == 0:
        return np.zeros((0, 3))
    return (keys + 0.5) * voxel


def latest_before(seq, t):
    best = None
    for item in seq:
        if item[0] <= t:
            best = item
        else:
            break
    return best


# ------------------------------------------------------------------------------------- plots
def takeovers(d):
    """times the tracker went from TRACKING / HOLD to DISABLED (pilot took over / left GUIDED).
    The status is only published at 1 Hz, so the time is refined to the last tracker reference
    message (published at 50 Hz while the tracker is enabled) before the DISABLED status."""
    out, prev = [], None
    ref_t = d['ref'][:, 0] if len(d['ref']) else np.zeros(0)
    for t, s in d['tstatus']:
        st = s.split()[0] if s else ''
        if st == 'DISABLED' and prev in ('TRACKING', 'HOLD'):
            before = ref_t[(ref_t <= t) & (ref_t > t - 1.5)]
            out.append(float(before[-1]) if len(before) else t)
        prev = st
    return out


def to_floor_frame(d, map_z_offset=None):
    """Bring everything to height above the real floor (the OptiTrack frame).
    The map, EKF pose, reference and plans are in the EKF map frame, which is shifted vertically
    from OptiTrack by map_mocap_z_offset (launch argument). Default: the measured mean EKF z -
    OptiTrack z of this bag. The EKF - OptiTrack difference is kept for the diagnostic panel."""
    moc, ekf = d['mocap'], d['ekf']
    d['dz'] = np.zeros((0, 2))
    if len(moc) > 1 and len(ekf) > 1:
        dz = np.interp(moc[:, 0], ekf[:, 0], ekf[:, 3]) - moc[:, 3]
        d['dz'] = np.column_stack([moc[:, 0], dz])
    if map_z_offset is None:
        map_z_offset = float(np.median(d['dz'][:, 1])) if len(d['dz']) else 0.0
    d['map_z_offset'] = map_z_offset
    if map_z_offset == 0.0:
        return
    for k in ('ekf', 'ref'):
        if len(d[k]):
            d[k][:, 3] -= map_z_offset
    d['clouds'] = [(t, p - np.array([0, 0, map_z_offset, 0])) if len(p) else (t, p) for t, p in d['clouds']]
    for k in ('plans', 'astar'):
        d[k] = [(t, p - np.array([0, 0, map_z_offset])) if len(p) else (t, p) for t, p in d[k]]
    d['yaw'] = [(t, p - np.array([0, 0, map_z_offset, 0])) if len(p) else (t, p) for t, p in d['yaw']]
    d['goals'] = [(t, x, y, z - map_z_offset) for t, x, y, z in d['goals']]


def fix_heading(d, mocap_yaw_offset):
    """The OptiTrack rigid-body axes are not the drone's (here x = left, -y = nose), so its yaw is
    not the heading. Use the EKF heading (base_link, x = nose) at the OptiTrack times when it is in
    the bag, otherwise add the rigid-body offset."""
    moc, ekf = d['mocap'], d['ekf']
    if not len(moc):
        return
    if len(ekf) > 1:
        yaw = np.unwrap(ekf[:, 4])
        moc[:, 4] = np.interp(moc[:, 0], ekf[:, 0], yaw)
    else:
        moc[:, 4] = moc[:, 4] + math.radians(mocap_yaw_offset)
    moc[:, 4] = np.arctan2(np.sin(moc[:, 4]), np.cos(moc[:, 4]))


def pose_at(arr, t):
    return np.array([np.interp(t, arr[:, 0], arr[:, k]) for k in (1, 2, 3, 4)])


def nearest_mapped(d, args):
    """per map message: distance from the drone to the nearest obstacle-surface point of THAT
    message (what the planner saw at the time), within +-band of the drone height"""
    path = d['mocap'] if len(d['mocap']) else d['ekf']
    out = []
    if len(path) < 2:
        return np.zeros((0, 5))
    for t, pts in d['clouds']:
        if len(pts) == 0 or t < path[0, 0] or t > path[-1, 0]:
            continue
        x = pose_at(path, t)[:3]
        sel = pts[(pts[:, 2] > args.floor_cut) & (pts[:, 3] <= args.surface) &
                  (np.abs(pts[:, 2] - x[2]) < args.band)]
        if len(sel) == 0:
            out.append((t, np.nan, np.nan, np.nan, np.nan))
            continue
        dd = np.linalg.norm(sel[:, :3] - x, axis=1)
        k = int(np.argmin(dd))
        out.append((t, dd[k], *sel[k, :3]))
    return np.array(out) if out else np.zeros((0, 5))


def plot_overview(d, args, out):
    import matplotlib.pyplot as plt

    m = accumulate_map(d['clouds'], args.voxel, args.floor_cut, args.surface, args.min_hits)
    ekf, moc, ref = d['ekf'], d['mocap'], d['ref']
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1])
    tko = takeovers(d)
    near = nearest_mapped(d, args)
    ax = fig.add_subplot(gs[:, 0])

    # map (top view, coloured by height)
    if len(m):
        sc = ax.scatter(m[:, 0], m[:, 1], c=m[:, 2], s=6, cmap='viridis', marker='s', linewidths=0,
                        vmin=args.floor_cut, vmax=max(args.floor_cut + 0.1, np.percentile(m[:, 2], 98)))
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, label='obstacle height above floor [m]')

    # planned trajectories: every replan faint, the last one solid
    plans = [(t, p) for t, p in d['plans'] if len(p) > 1]
    if plans and not args.no_plans:
        for _, p in plans[:-1]:
            ax.plot(p[:, 0], p[:, 1], color='tab:orange', lw=0.9, alpha=0.3)
        ax.plot(plans[-1][1][:, 0], plans[-1][1][:, 1], color='tab:orange', lw=1.8)
    if args.astar:
        for _, p in d['astar']:
            if len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], color='0.6', lw=0.6, ls=':', alpha=0.6)

    if len(moc):
        ax.plot(moc[:, 1], moc[:, 2], color='tab:blue', lw=2.0, label='flown (OptiTrack)')
    if len(ekf):
        ax.plot(ekf[:, 1], ekf[:, 2], color='k', lw=1.2, ls='--', label='flown (EKF)')
    if len(ref) and args.ref:
        ax.plot(ref[:, 1], ref[:, 2], color='tab:green', lw=1.0, label='tracker reference')
    path = moc if len(moc) else ekf
    if len(path):
        ax.plot(path[0, 1], path[0, 2], 'o', color='tab:green', ms=9, mec='k', label='start', zorder=5)
        if args.heading:
            step = max(1, len(path) // 25)
            ax.quiver(path[::step, 1], path[::step, 2], np.cos(path[::step, 4]), np.sin(path[::step, 4]),
                      color='tab:blue', scale=25, width=0.004, alpha=0.8, zorder=4)
    for i, (t, gx, gy, gz) in enumerate(d['goals']):
        ax.plot(gx, gy, '*', color='tab:red', ms=16, mec='k', zorder=6, label='goal' if i == 0 else None)
    if plans and not args.no_plans:
        ax.plot([], [], color='tab:orange', lw=1.5, label=f'planned B-splines ({len(plans)})')
    path_t = moc if len(moc) > 1 else ekf
    for i, tk in enumerate(tko):
        if len(path_t) < 2:
            break
        x = pose_at(path_t, tk)
        ax.plot(x[0], x[1], 'X', color='tab:red', ms=12, mec='k', zorder=7,
                label='pilot took over' if i == 0 else None)
        if len(near):
            j = int(np.argmin(np.abs(near[:, 0] - tk)))
            if np.isfinite(near[j, 1]):
                ax.plot([x[0], near[j, 2]], [x[1], near[j, 3]], color='tab:red', lw=1.5, ls='--', zorder=7)
                ax.annotate(f'map: {near[j, 1]:.2f} m', (x[0], x[1]), textcoords='offset points',
                            xytext=(8, -14), color='tab:red', fontsize=9, weight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Top view: map ({d["map_topic"] or "none"}), flown path and plans')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    # side view along the flight: height vs distance flown
    ax2 = fig.add_subplot(gs[0, 1])
    for arr, lab, sty in ((moc, 'OptiTrack', dict(color='tab:blue', lw=2)),
                          (ekf, 'EKF', dict(color='k', lw=1, ls='--')),
                          (ref, 'reference', dict(color='tab:green', lw=1))):
        if len(arr):
            ax2.plot(arr[:, 0], arr[:, 3], label=lab, **sty)
    ax2.set_ylabel('z [m]')
    ax2.set_title(f'Height above floor (EKF, reference, map: {-d["map_z_offset"]:+.2f} m applied)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # tracking error: reference vs EKF (what the controller saw) and vs OptiTrack (truth)
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    for arr, lab, col in ((ekf, 'reference - EKF', 'k'), (moc, 'reference - OptiTrack', 'tab:blue')):
        if len(arr) > 1 and len(ref) > 1:
            px = np.interp(ref[:, 0], arr[:, 0], arr[:, 1])
            py = np.interp(ref[:, 0], arr[:, 0], arr[:, 2])
            pz = np.interp(ref[:, 0], arr[:, 0], arr[:, 3])
            err = np.sqrt((ref[:, 1] - px) ** 2 + (ref[:, 2] - py) ** 2 + (ref[:, 3] - pz) ** 2)
            ax3.plot(ref[:, 0], err, color=col, lw=1, label=f'{lab} (mean {err.mean():.2f} m)')
    if len(d['dz']):
        ex = d['dz'][:, 1]
        ax3.plot(d['dz'][:, 0], ex, color='tab:purple', lw=0.8, alpha=0.7,
                 label=f'EKF z - OptiTrack z (mean {ex.mean():+.2f} m, map shifted by '
                       f'{d["map_z_offset"]:+.2f})')
    ax3.set_ylabel('[m]')
    ax3.set_title('Tracking error')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    # speed + yaw, with tracker HOLD / DISABLED periods shaded
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)
    src = moc if len(moc) > 2 else ekf
    if len(src) > 2:
        dt = np.gradient(src[:, 0])
        dt[dt <= 0] = np.nan
        sp = np.sqrt(np.gradient(src[:, 1]) ** 2 + np.gradient(src[:, 2]) ** 2 + np.gradient(src[:, 3]) ** 2) / dt
        ax4.plot(src[:, 0], sp, color='tab:blue', lw=1, label='speed [m/s]')
        ax4b = ax4.twinx()
        ax4b.plot(src[:, 0], np.degrees(np.unwrap(src[:, 4])), color='tab:orange', lw=1, label='yaw [deg]')
        ax4b.set_ylabel('yaw [deg]', color='tab:orange')
    shade_status(ax4, d['tstatus'])
    ax4.set_ylabel('speed [m/s]', color='tab:blue')
    ax4.set_title('Speed and yaw (shaded: tracker HOLD red, DISABLED grey)')
    ax4.grid(True, alpha=0.3)

    # what the map said: nearest obstacle surface in each map message, near the drone's height
    ax5 = fig.add_subplot(gs[3, 1], sharex=ax2)
    if len(near):
        ax5.plot(near[:, 0], near[:, 1], color='tab:red', lw=1, marker='.', ms=2)
    ax5.set_ylim(bottom=0)
    ax5.set_xlabel('time [s]')
    ax5.set_ylabel('[m]')
    ax5.set_title(f'Nearest mapped obstacle (latest map, within +-{args.band:.1f} m of drone height)')
    ax5.grid(True, alpha=0.3)
    for a in (ax2, ax3, ax4, ax5):
        for tk in tko:
            a.axvline(tk, color='tab:red', lw=1.2, ls='--')
    for tk in tko:
        x = pose_at(path_t, tk) if len(path_t) > 1 else None
        msg = f'  pilot took over at t = {tk:.1f} s'
        if x is not None:
            msg += f', drone at ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})'
        if len(near):
            j = int(np.argmin(np.abs(near[:, 0] - tk)))
            msg += f', nearest mapped obstacle {near[j, 1]:.2f} m'
        print(msg)
        seen = set()
        for t, st in d['pstatus']:
            if tk - 6.0 <= t <= tk + 1.0 and st not in seen:
                seen.add(st)
                print(f'    {t:6.1f} s planner: {st}')

    print_summary(d, m)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'saved {out}')
    return fig


def shade_status(ax, status):
    if not status:
        return
    cur, start = None, None
    for t, s in status + [(status[-1][0] + 1.0, 'END')]:
        st = s.split()[0] if s else ''
        if st != cur:
            if cur in ('HOLD', 'DISABLED') and start is not None:
                ax.axvspan(start, t, color='tab:red' if cur == 'HOLD' else '0.7', alpha=0.15, lw=0)
            cur, start = st, t


def plot_snapshots(d, args, out_prefix):
    import matplotlib.pyplot as plt
    plans = [(t, p) for t, p in d['plans'] if len(p) > 1]
    path = d['mocap'] if len(d['mocap']) else d['ekf']
    if not plans or not len(path):
        print('snapshots need plans and a pose topic')
        return
    t_lo, t_hi = plans[0][0], plans[-1][0]
    times = np.linspace(t_lo, t_hi, args.snapshots)
    cols = min(3, args.snapshots)
    rows = int(math.ceil(args.snapshots / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    xs = np.concatenate([path[:, 1]] + [p[:, 0] for _, p in plans])
    ys = np.concatenate([path[:, 2]] + [p[:, 1] for _, p in plans])
    pad = 0.5
    for ax, t in zip(axs.ravel(), times):
        # map as seen up to t (no hit filter on the latest cloud so new obstacles show)
        m = accumulate_map(d['clouds'], args.voxel, args.floor_cut, args.surface, args.min_hits, t_max=t)
        last = latest_before(d['clouds'], t)
        if len(m):
            ax.scatter(m[:, 0], m[:, 1], c='0.55', s=3, marker='s', linewidths=0)
        if last is not None and len(last[1]):
            p = last[1]
            p = p[(p[:, 2] > args.floor_cut) & (p[:, 3] <= args.surface)]
            ax.scatter(p[:, 0], p[:, 1], c='k', s=3, marker='s', linewidths=0)
        pl = latest_before(plans, t)
        if pl is not None:
            ax.plot(pl[1][:, 0], pl[1][:, 1], color='tab:orange', lw=2, label='active plan')
        yw = latest_before(d['yaw'], t)
        if yw is not None and len(yw[1]):
            a = yw[1]
            ax.quiver(a[:, 0], a[:, 1], np.cos(a[:, 3]), np.sin(a[:, 3]), color='tab:purple',
                      scale=18, width=0.006, label='planned yaw')
        flown = path[path[:, 0] <= t]
        ax.plot(flown[:, 1], flown[:, 2], color='tab:blue', lw=1.5, label='flown')
        if len(flown):
            ax.plot(flown[-1, 1], flown[-1, 2], 'o', color='tab:blue', ms=7, mec='k')
            ax.arrow(flown[-1, 1], flown[-1, 2], 0.4 * math.cos(flown[-1, 4]), 0.4 * math.sin(flown[-1, 4]),
                     width=0.02, color='tab:blue')
        g = latest_before(d['goals'], t)
        if g is not None:
            ax.plot(g[1], g[2], '*', color='tab:red', ms=14, mec='k')
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.1f} s')
        ax.grid(True, alpha=0.3)
    axs.ravel()[0].legend(fontsize=7, loc='best')
    for ax in axs.ravel()[len(times):]:
        ax.axis('off')
    fig.suptitle('Map (grey: accumulated, black: latest), active plan and flown path')
    fig.tight_layout()
    out = f'{out_prefix}_snapshots.{args.fmt}'
    fig.savefig(out, dpi=200)
    print(f'saved {out}')
    return fig


def plot_takeover(d, args, out_prefix):
    """the map as the planner had it when the pilot took over, zoomed on the drone"""
    import matplotlib.pyplot as plt
    tko = takeovers(d)
    path = d['mocap'] if len(d['mocap']) > 1 else d['ekf']
    if not tko or len(path) < 2:
        return None
    near = nearest_mapped(d, args)
    n = len(tko)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 6.5), squeeze=False)
    for ax, tk in zip(axs[0], tko):
        x = pose_at(path, tk)
        m = accumulate_map(d['clouds'], args.voxel, args.floor_cut, args.surface, args.min_hits, t_max=tk)
        if len(m):
            band = np.abs(m[:, 2] - x[2]) < args.band
            ax.scatter(m[~band, 0], m[~band, 1], c='0.75', s=10, marker='s', linewidths=0,
                       label='mapped (other heights)')
            ax.scatter(m[band, 0], m[band, 1], c='0.25', s=10, marker='s', linewidths=0,
                       label=f'mapped (drone height +-{args.band:.1f} m)')
        last = latest_before(d['clouds'], tk)
        if last is not None and len(last[1]):
            q = last[1]
            q = q[(q[:, 2] > args.floor_cut) & (q[:, 3] <= args.surface) & (np.abs(q[:, 2] - x[2]) < args.band)]
            ax.scatter(q[:, 0], q[:, 1], c='tab:red', s=6, marker='s', linewidths=0, alpha=0.6,
                       label='latest map message')
        pl = latest_before([(t, p) for t, p in d['plans'] if len(p) > 1], tk)
        if pl is not None:
            ax.plot(pl[1][:, 0], pl[1][:, 1], color='tab:orange', lw=2, label='active plan')
        flown = path[path[:, 0] <= tk + 3.0]
        ax.plot(flown[:, 1], flown[:, 2], color='tab:blue', lw=1.5, label='flown (incl. 3 s after)')
        ax.plot(x[0], x[1], 'X', color='tab:red', ms=13, mec='k', zorder=6, label='take-over')
        ax.arrow(x[0], x[1], 0.5 * math.cos(x[3]), 0.5 * math.sin(x[3]), width=0.03, color='tab:blue', zorder=6)
        hf = math.radians(args.hfov / 2)
        for s_ in (-1, 1):                                    # camera FOV wedge, 4 m
            ax.plot([x[0], x[0] + 4 * math.cos(x[3] + s_ * hf)], [x[1], x[1] + 4 * math.sin(x[3] + s_ * hf)],
                    color='tab:blue', lw=0.8, ls=':')
        if len(near):
            j = int(np.argmin(np.abs(near[:, 0] - tk)))
            if np.isfinite(near[j, 1]):
                ax.plot([x[0], near[j, 2]], [x[1], near[j, 3]], 'r--', lw=1.5)
                ax.set_title(f'take-over t = {tk:.1f} s: nearest mapped obstacle {near[j, 1]:.2f} m')
        ax.set_xlim(x[0] - args.zoom, x[0] + args.zoom)
        ax.set_ylim(x[1] - args.zoom, x[1] + args.zoom)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    out = f'{out_prefix}_takeover.{args.fmt}'
    fig.savefig(out, dpi=200)
    print(f'saved {out}')
    return fig


def plot_3d(d, args, out_prefix):
    import matplotlib.pyplot as plt
    m = accumulate_map(d['clouds'], args.voxel, args.floor_cut, args.surface, args.min_hits)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    if len(m):
        ax.scatter(m[:, 0], m[:, 1], m[:, 2], c=m[:, 2], s=2, cmap='viridis', alpha=0.5)
    for _, p in d['plans']:
        if len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color='tab:orange', alpha=0.25, lw=0.8)
    path = d['mocap'] if len(d['mocap']) else d['ekf']
    if len(path):
        ax.plot(path[:, 1], path[:, 2], path[:, 3], color='tab:blue', lw=2.5, label='flown')
    for _, gx, gy, gz in d['goals'][:1]:
        ax.scatter([gx], [gy], [gz], marker='*', s=200, color='tab:red', label='goal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    lim = np.array([ax.get_xlim3d(), ax.get_ylim3d()])
    r = (lim[:, 1] - lim[:, 0]).max() / 2
    c = lim.mean(axis=1)
    ax.set_xlim3d(c[0] - r, c[0] + r)
    ax.set_ylim3d(c[1] - r, c[1] + r)
    out = f'{out_prefix}_3d.{args.fmt}'
    fig.savefig(out, dpi=200)
    print(f'saved {out}')
    return fig


def print_summary(d, m):
    path = d['mocap'] if len(d['mocap']) else d['ekf']
    if len(path) > 1:
        L = np.sum(np.linalg.norm(np.diff(path[:, 1:4], axis=0), axis=1))
        print(f'  flown: {L:.2f} m in {path[-1, 0] - path[0, 0]:.1f} s')
    print(f'  plans: {len(d["plans"])}   map clouds: {len(d["clouds"])}   map voxels drawn: {len(m)}')
    if d['goals']:
        g = d['goals'][-1]
        if len(path):
            print(f'  final distance to goal: {math.dist(path[-1, 1:4], g[1:4]):.2f} m')
    if d['mocap_frame']:
        print(f'  note: OptiTrack pose frame "{d["mocap_frame"]}" is drawn as if equal to map')
    for t, s in d['pstatus']:
        if 'EMERGENCY' in s:
            print(f'  {t:6.1f} s  planner: {s}')
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='rosbag2 directory (or .mcap / .db3 file)')
    ap.add_argument('--out', help='output file prefix (default: <bag>)')
    ap.add_argument('--t0', type=float, help='start time [s from bag start]')
    ap.add_argument('--t1', type=float, help='end time [s from bag start]')
    ap.add_argument('--voxel', type=float, default=0.05, help='map voxel [m]')
    ap.add_argument('--floor-cut', type=float, default=0.15, help='hide map points below this z [m]')
    ap.add_argument('--surface', type=float, default=0.10,
                    help='keep map points with ESDF <= this (obstacle surfaces) [m]')
    ap.add_argument('--min-hits', type=int, default=3, help='voxel must appear in this many clouds')
    ap.add_argument('--zoom', type=float, default=3.0, help='take-over plot half width [m]')
    ap.add_argument('--map-z-offset', type=float, default=None,
                    help='map_mocap_z_offset used when flying [m] (default: measured EKF - OptiTrack '
                         'median); map, EKF, reference and plans are shifted down by it so all heights '
                         'are above the real floor')
    ap.add_argument('--mocap-yaw-offset', type=float, default=-90.0,
                    help='heading = OptiTrack yaw + this [deg], only used without EKF pose '
                         '(rigid body dilon: x = left, -y = nose -> -90)')
    ap.add_argument('--hfov', type=float, default=60.0, help='camera horizontal FOV for the wedge [deg]')
    ap.add_argument('--band', type=float, default=0.3,
                    help='nearest-obstacle panel: only map points within +-band of the drone height [m]')
    ap.add_argument('--snapshots', type=int, default=0, help='also plot N snapshots of map + active plan')
    ap.add_argument('--3d', dest='three_d', action='store_true', help='also a 3D plot')
    ap.add_argument('--astar', action='store_true', help='draw the A* paths too')
    ap.add_argument('--no-plans', action='store_true', help='do not draw the planned B-splines')
    ap.add_argument('--no-ref', dest='ref', action='store_false', help='do not draw the tracker reference')
    ap.add_argument('--no-heading', dest='heading', action='store_false', help='no heading arrows')
    ap.add_argument('--pdf', action='store_true', help='save PDF instead of PNG (for LaTeX)')
    ap.add_argument('--show', action='store_true', help='open the figures')
    ap.add_argument('--mcap', action='store_true',
                    help='read the .mcap file(s) directly (skip metadata.yaml / rosbags reader)')
    args = ap.parse_args()
    args.fmt = 'pdf' if args.pdf else 'png'
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')

    d = read_bag(args.bag, args.t0, args.t1, args.surface, args.mcap)
    fix_heading(d, args.mocap_yaw_offset)
    to_floor_frame(d, args.map_z_offset)
    print(f'  map frame -> floor frame: z - {d["map_z_offset"]:.3f} m '
          f'({"--map-z-offset" if args.map_z_offset is not None else "measured EKF - OptiTrack"})')
    prefix = args.out or str(Path(args.bag).with_suffix('')).rstrip('/')
    plot_overview(d, args, f'{prefix}_overview.{args.fmt}')
    plot_takeover(d, args, prefix)
    if args.snapshots > 0:
        plot_snapshots(d, args, prefix)
    if args.three_d:
        plot_3d(d, args, prefix)
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
