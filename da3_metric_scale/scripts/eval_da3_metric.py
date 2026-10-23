#!/usr/bin/env python3
"""Offline evaluation of da3_metric_scaler output from a rosbag.

Record a run first, e.g.:
  ros2 bag record /da3_metric/scale /da3_metric/depth /da3_metric/camera_info /tf \
      /mavros/dilon/pose_throttled [/vrpn_mocap/<target>/pose] -o da3_eval_run1

Then:
  python3 eval_da3_metric.py da3_eval_run1
  python3 eval_da3_metric.py da3_eval_run1 --target-topic /vrpn_mocap/board/pose

Checks
  1. Scale stability   : fused / floor / triangulation scale over time, agreement.
  2. Floor consistency : metric depth of floor pixels vs. depth predicted by the mocap
                         camera pose and the floor plane (ratio should be 1.0, floor
                         height 0). Independent only if the run used
                         method:=triangulation (otherwise the floor was fitted on).
  3. Target (optional) : a separate mocap rigid body (board/box). Its true camera-frame
                         depth vs. metric depth at its projected pixel. Fully independent
                         ground truth -> AbsRel, RMSE, delta<1.25 per depth bin.

Outputs: printed summary, PNG plots and CSVs in --out (default <bag>_eval/).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# Geometry helpers (pure numpy)
# ----------------------------------------------------------------------------


def quat_to_rot(q):
    """q = (x, y, z, w) -> 3x3."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, s):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


class PoseTrack:
    """Time-sorted poses (t, p[3], q[4] xyzw) with interpolation."""

    def __init__(self, t, p, q):
        order = np.argsort(t)
        self.t = np.asarray(t, float)[order]
        self.p = np.asarray(p, float)[order]
        self.q = np.asarray(q, float)[order]

    def __len__(self):
        return len(self.t)

    def at(self, t, max_gap=0.1, exact_tol=1e-4):
        if len(self.t) == 0:
            return None
        i = int(np.searchsorted(self.t, t))
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= exact_tol:
                return self.p[j], quat_to_rot(self.q[j])
        if i == 0 or i >= len(self.t):
            return None
        t0, t1 = self.t[i - 1], self.t[i]
        if t1 - t0 > max_gap:
            return None
        s = (t - t0) / (t1 - t0)
        p = (1 - s) * self.p[i - 1] + s * self.p[i]
        return p, quat_to_rot(slerp(self.q[i - 1], self.q[i], s))


def undistort_points(uv, K, D, iters=10):
    """Pixel coords (N,2) -> normalized (N,2) for plumb_bob/radtan (k1 k2 p1 p2 [k3])."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    d = np.zeros(5)
    d[:min(5, len(D))] = D[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    return np.stack([x, y], axis=1)


def distort_project(pc, K, D):
    """Camera-frame points (N,3), z>0 -> pixels (N,2) with radtan distortion."""
    x = pc[:, 0] / pc[:, 2]
    y = pc[:, 1] / pc[:, 2]
    d = np.zeros(5)
    d[:min(5, len(D))] = D[:5]
    k1, k2, p1, p2, k3 = d
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)


# ----------------------------------------------------------------------------
# Per-frame analyses
# ----------------------------------------------------------------------------


class FloorGrid:
    def __init__(self, K, D, w, h, roi_frac=0.5, step=8):
        v0 = int(h * (1.0 - roi_frac))
        vs, us = np.mgrid[v0 + step // 2:h:step, step // 2:w:step]
        self.u = us.ravel().astype(int)
        self.v = vs.ravel().astype(int)
        n = undistort_points(np.stack([self.u, self.v], 1).astype(float), K, D)
        self.rays = np.concatenate([n, np.ones((len(n), 1))], axis=1)  # z = 1


def floor_frame(depth, grid, c, R, up, floor_h, min_graze_deg=15.0, max_range=6.0,
                ratio_band=(0.5, 2.0)):
    """Returns dict of per-pixel arrays for valid floor samples, or None."""
    h = float(up @ c) - floor_h
    if h < 0.15:
        return None
    rw = grid.rays @ R.T                                  # world ray dirs (z_cam = 1)
    denom = rw @ up
    cosg = -denom / np.linalg.norm(rw, axis=1)
    ok = (denom < 0) & (cosg > math.sin(math.radians(min_graze_deg)))
    z_geo = np.full(len(denom), np.nan)
    z_geo[ok] = h / -denom[ok]
    ok &= (z_geo > 0.1) & (z_geo < max_range)
    z_m = depth[grid.v, grid.u].astype(float)
    ok &= np.isfinite(z_m) & (z_m > 0)
    if ok.sum() < 20:
        return None
    ratio = z_m[ok] / z_geo[ok]
    band = (ratio > ratio_band[0]) & (ratio < ratio_band[1])  # drop obvious occluders
    if band.sum() < 20:
        return None
    zg, zm, rt, dn = z_geo[ok][band], z_m[ok][band], ratio[band], denom[ok][band]
    height = h + zm * dn                                   # height of metric point
    return {'z_geo': zg, 'z_m': zm, 'ratio': rt, 'height': height}


def target_frame(depth, K, D, c, R, p_target_w, win=3, min_z=0.2):
    pc = R.T @ (p_target_w - c)
    if pc[2] < min_z:
        return None
    uv = distort_project(pc[None, :], K, D)[0]
    u, v = int(round(uv[0])), int(round(uv[1]))
    H, W = depth.shape
    if u < win or v < win or u >= W - win or v >= H - win:
        return None
    patch = depth[v - win:v + win + 1, u - win:u + win + 1].astype(float)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size < 0.3 * (2 * win + 1) ** 2:
        return None
    return {'z_true': float(pc[2]), 'z_pred': float(np.median(patch)), 'u': u, 'v': v}


def depth_metrics(z_true, z_pred):
    z_true, z_pred = np.asarray(z_true), np.asarray(z_pred)
    if len(z_true) == 0:
        return None
    r = np.maximum(z_pred / z_true, z_true / z_pred)
    return {
        'n': len(z_true),
        'absrel': float(np.mean(np.abs(z_pred - z_true) / z_true)),
        'rmse': float(np.sqrt(np.mean((z_pred - z_true) ** 2))),
        'bias': float(np.median(z_pred / z_true) - 1.0),
        'd1': float(np.mean(r < 1.25)),
        'd105': float(np.mean(r < 1.05)),
    }


# ----------------------------------------------------------------------------
# Bag reading
# ----------------------------------------------------------------------------


def open_reader(bag, topics):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in topics if t in types]
    if present:
        reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    return reader, types


def iter_msgs(bag, topics):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader, types = open_reader(bag, topics)
    cls = {t: get_message(types[t]) for t in topics if t in types}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in cls:
            yield topic, deserialize_message(data, cls[topic]), t_ns * 1e-9


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def image_to_array(msg):
    if msg.encoding not in ('32FC1', '32FC'):
        raise ValueError(f'expected 32FC1 depth, got {msg.encoding}')
    dt = np.dtype('<f4') if not msg.is_bigendian else np.dtype('>f4')
    a = np.frombuffer(bytes(msg.data), dtype=dt)
    return a.reshape(msg.height, msg.step // 4)[:, :msg.width]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def summarize(name, x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return f'{name}: no data'
    med = np.median(x)
    return (f'{name}: n={len(x)} median={med:.4f} mean={x.mean():.4f} std={x.std():.4f} '
            f'CV={x.std() / abs(med) * 100:.2f}% p5={np.percentile(x, 5):.4f} '
            f'p95={np.percentile(x, 95):.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--out', default=None)
    ap.add_argument('--scale-topic', default='/da3_metric/scale')
    ap.add_argument('--depth-topic', default='/da3_metric/depth')
    ap.add_argument('--info-topic', default='/da3_metric/camera_info')
    ap.add_argument('--world-frame', default='mocap')
    ap.add_argument('--camera-frame', default='camera_optical')
    ap.add_argument('--world-up', type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument('--floor-height', type=float, default=0.0)
    ap.add_argument('--floor-roi', type=float, default=0.5,
                    help='bottom fraction of image used for the floor check')
    ap.add_argument('--target-topic', default=None,
                    help='PoseStamped of a separate mocap rigid body (ground truth)')
    ap.add_argument('--target-offset', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help='point on the target surface in its body frame (m)')
    ap.add_argument('--target-time-offset', type=float, default=0.0,
                    help='s added to depth stamps when looking up the target pose')
    ap.add_argument('--every', type=int, default=1, help='use every Nth depth frame')
    ap.add_argument('--bins', type=float, nargs='+', default=[0, 1, 2, 3, 4, 6])
    args = ap.parse_args()

    bag = args.bag.rstrip('/')
    out = args.out or (bag + '_eval')
    os.makedirs(out, exist_ok=True)
    up = np.array(args.world_up) / np.linalg.norm(args.world_up)

    # ---------------- pass 1: scale, tf, camera_info, target ----------------
    scale_rows = []          # t, fused_s, fused_b, floor_s, floor_n, tri_s, tri_n, edge
    tf_t, tf_p, tf_q = [], [], []
    tg_t, tg_p, tg_q = [], [], []
    info = None
    topics = [args.scale_topic, '/tf', args.info_topic]
    if args.target_topic:
        topics.append(args.target_topic)
    for topic, msg, t_bag in iter_msgs(bag, topics):
        if topic == args.scale_topic:
            d = list(msg.data) + [math.nan] * (7 - len(msg.data))
            scale_rows.append([t_bag] + d[:7])
        elif topic == '/tf':
            for tr in msg.transforms:
                if (tr.header.frame_id.lstrip('/') == args.world_frame and
                        tr.child_frame_id.lstrip('/') == args.camera_frame):
                    a, r = tr.transform.translation, tr.transform.rotation
                    tf_t.append(stamp(tr.header))
                    tf_p.append([a.x, a.y, a.z])
                    tf_q.append([r.x, r.y, r.z, r.w])
        elif topic == args.info_topic and info is None:
            info = msg
        elif topic == args.target_topic:
            a, r = msg.pose.position, msg.pose.orientation
            tg_t.append(stamp(msg.header))
            tg_p.append([a.x, a.y, a.z])
            tg_q.append([r.x, r.y, r.z, r.w])

    cam = PoseTrack(tf_t, tf_p, tf_q)
    tgt = PoseTrack(tg_t, tg_p, tg_q) if tg_t else None
    print(f'bag: {bag}')
    print(f'  scale msgs: {len(scale_rows)}  camera TF: {len(cam)}  '
          f'target poses: {len(tg_t) if tg_t else 0}  camera_info: {info is not None}')

    # ---------------- 1. scale stability ----------------
    print('\n[1] Scale stability')
    S = np.array(scale_rows, float) if scale_rows else np.zeros((0, 8))
    if len(S):
        t0 = S[0, 0]
        print('  ' + summarize('fused s', S[:, 1]))
        print('  ' + summarize('floor s', S[:, 3]))
        print('  ' + summarize('tri   s', S[:, 5]))
        both = np.isfinite(S[:, 3]) & np.isfinite(S[:, 5])
        if both.any():
            print('  ' + summarize('floor/tri ratio', S[both, 3] / S[both, 5]))
        print('  ' + summarize('edge fraction', S[:, 7]))
        with open(os.path.join(out, 'scale.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'fused_s', 'fused_b', 'floor_s', 'floor_n', 'tri_s', 'tri_n',
                        'edge_frac'])
            w.writerows(S.tolist())
    else:
        print('  no scale messages')

    if info is None:
        print('\nNo camera_info in bag - cannot run depth checks.')
        plot_all(out, S, None, None, args.bins)
        return
    if len(cam) == 0:
        print(f'\nNo /tf {args.world_frame} -> {args.camera_frame} in bag - '
              'cannot run depth checks (publish_tf must be true when recording).')
        plot_all(out, S, None, None, args.bins)
        return

    K = np.array(info.k, float).reshape(3, 3)
    D = np.array(info.d, float)
    if info.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
        print(f'  warning: distortion model {info.distortion_model} treated as radtan')
    grid = FloorGrid(K, D, info.width, info.height, roi_frac=args.floor_roi)

    # ---------------- pass 2: depth frames ----------------
    floor_frames, floor_px = [], {'z_geo': [], 'z_m': [], 'ratio': [], 'height': []}
    tgt_rows = []
    n_depth = n_pose = 0
    for k, (topic, msg, _) in enumerate(iter_msgs(bag, [args.depth_topic])):
        if k % args.every:
            continue
        n_depth += 1
        t = stamp(msg.header)
        pose = cam.at(t)
        if pose is None:
            continue
        n_pose += 1
        c, R = pose
        depth = image_to_array(msg)
        if depth.shape != (info.height, info.width):
            print(f'  depth {depth.shape} != camera_info {(info.height, info.width)}; skipping')
            continue

        fl = floor_frame(depth, grid, c, R, up, args.floor_height)
        if fl is not None:
            floor_frames.append([t, np.median(fl['ratio']), np.median(fl['height']),
                                 float(up @ c) - args.floor_height, len(fl['ratio'])])
            for key in floor_px:
                floor_px[key].append(fl[key])

        if tgt is not None:
            tp = tgt.at(t + args.target_time_offset)
            if tp is not None:
                p_w = tp[1] @ np.array(args.target_offset) + tp[0]
                r = target_frame(depth, K, D, c, R, p_w)
                if r is not None:
                    tgt_rows.append([t, r['z_true'], r['z_pred'], r['u'], r['v']])

    print(f'\n  depth frames used: {n_depth}, with camera pose: {n_pose}')

    # ---------------- 2. floor ----------------
    print('\n[2] Floor consistency (metric depth / geometric floor depth; ideal 1.0)')
    F = np.array(floor_frames, float) if floor_frames else np.zeros((0, 5))
    FP = {k: np.concatenate(v) if v else np.zeros(0) for k, v in floor_px.items()}
    if len(F):
        print('  ' + summarize('per-frame median ratio', F[:, 1]))
        print('  ' + summarize('per-frame median floor height [m]', F[:, 2]))
        rt = FP['ratio']
        print(f'  pixels: n={len(rt)}  within 5%: {np.mean(np.abs(rt - 1) < 0.05) * 100:.1f}%  '
              f'within 10%: {np.mean(np.abs(rt - 1) < 0.10) * 100:.1f}%')
        print('  by geometric depth:')
        for lo, hi in zip(args.bins[:-1], args.bins[1:]):
            m = (FP['z_geo'] >= lo) & (FP['z_geo'] < hi)
            if m.sum() < 50:
                continue
            met = depth_metrics(FP['z_geo'][m], FP['z_m'][m])
            print(f'    {lo:.0f}-{hi:.0f} m: n={met["n"]:7d} AbsRel={met["absrel"]:.3f} '
                  f'RMSE={met["rmse"]:.3f} m bias={met["bias"] * 100:+.1f}% '
                  f'd<1.05={met["d105"] * 100:.1f}% d<1.25={met["d1"] * 100:.1f}%')
        with open(os.path.join(out, 'floor_frames.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'median_ratio', 'median_height_m', 'cam_height_m', 'n_px'])
            w.writerows(F.tolist())
        print('  NOTE: only independent if the run used method:=triangulation.')
    else:
        print('  no usable floor frames (camera too low, floor not visible, or no pose)')

    # ---------------- 3. target ----------------
    T = np.array(tgt_rows, float) if tgt_rows else np.zeros((0, 5))
    if args.target_topic:
        print(f'\n[3] Target ground truth ({args.target_topic})')
        if len(T):
            met = depth_metrics(T[:, 1], T[:, 2])
            print(f'  all: n={met["n"]} AbsRel={met["absrel"]:.3f} RMSE={met["rmse"]:.3f} m '
                  f'bias={met["bias"] * 100:+.1f}% d<1.05={met["d105"] * 100:.1f}% '
                  f'd<1.25={met["d1"] * 100:.1f}%')
            for lo, hi in zip(args.bins[:-1], args.bins[1:]):
                m = (T[:, 1] >= lo) & (T[:, 1] < hi)
                if m.sum() < 5:
                    continue
                mb = depth_metrics(T[m, 1], T[m, 2])
                print(f'    {lo:.0f}-{hi:.0f} m: n={mb["n"]:5d} AbsRel={mb["absrel"]:.3f} '
                      f'RMSE={mb["rmse"]:.3f} m bias={mb["bias"] * 100:+.1f}%')
            with open(os.path.join(out, 'target.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t', 'z_true', 'z_pred', 'u', 'v'])
                w.writerows(T.tolist())
        else:
            print('  target never visible in a frame with valid depth')

    plot_all(out, S, (F, FP) if len(F) else None, T if len(T) else None, args.bins)
    print(f'\nplots + CSVs written to {out}/')


def plot_all(out, S, floor, T, bins):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available - skipping plots')
        return

    if len(S):
        t = S[:, 0] - S[0, 0]
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(t, S[:, 1], label='fused', lw=1.8)
        ax[0].plot(t, S[:, 3], '.', ms=3, label='floor')
        ax[0].plot(t, S[:, 5], '.', ms=3, label='triangulation')
        ax[0].set_ylabel('scale s')
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(t, S[:, 4], label='floor inliers')
        ax[1].plot(t, S[:, 6], label='tri inliers')
        ax[1].set_ylabel('inliers')
        ax[1].set_xlabel('time [s]')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'scale.png'), dpi=150)
        plt.close(fig)

    if floor is not None:
        F, FP = floor
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(F[:, 0] - F[0, 0], F[:, 1])
        ax[0].axhline(1.0, color='k', lw=0.8)
        ax[0].set_xlabel('time [s]')
        ax[0].set_ylabel('median metric / geometric depth')
        ax[0].set_title('Floor ratio per frame')
        ax[0].grid(alpha=0.3)
        centers, med, p25, p75 = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (FP['z_geo'] >= lo) & (FP['z_geo'] < hi)
            if m.sum() < 50:
                continue
            r = FP['ratio'][m]
            centers.append(0.5 * (lo + hi))
            med.append(np.median(r))
            p25.append(np.percentile(r, 25))
            p75.append(np.percentile(r, 75))
        if centers:
            med = np.array(med)
            ax[1].errorbar(centers, med, yerr=[med - np.array(p25), np.array(p75) - med],
                           fmt='o-', capsize=4)
        ax[1].axhline(1.0, color='k', lw=0.8)
        ax[1].set_xlabel('geometric floor depth [m]')
        ax[1].set_ylabel('ratio (median, IQR)')
        ax[1].set_title('Floor ratio vs depth')
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'floor.png'), dpi=150)
        plt.close(fig)

    if T is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lim = [0, max(T[:, 1].max(), T[:, 2].max()) * 1.05]
        ax[0].plot(T[:, 1], T[:, 2], '.', ms=4)
        ax[0].plot(lim, lim, 'k-', lw=0.8)
        ax[0].set_xlim(lim)
        ax[0].set_ylim(lim)
        ax[0].set_xlabel('true depth (mocap) [m]')
        ax[0].set_ylabel('metric DA3 depth [m]')
        ax[0].set_title('Target depth')
        ax[0].grid(alpha=0.3)
        ax[1].plot(T[:, 0] - T[0, 0], (T[:, 2] - T[:, 1]) / T[:, 1] * 100, '.', ms=4)
        ax[1].axhline(0, color='k', lw=0.8)
        ax[1].set_xlabel('time [s]')
        ax[1].set_ylabel('relative error [%]')
        ax[1].set_title('Target error over time')
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'target.png'), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
