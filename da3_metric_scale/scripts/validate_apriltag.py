#!/usr/bin/env python3
"""Validate metric DA3 depth against an AprilTag grid (Kalibr aprilgrid).

The grid gives an independent metric ground truth: tags are detected in the raw
image, the camera->board pose is solved with PnP from the known grid geometry, and
the true depth of the board plane is compared with /da3_metric/depth at every pixel
inside the board. No mocap and no floor involved.

Record (hold the grid in view at different distances, ~0.3-2 m, and angles):
  ros2 bag record /image_throttled /da3_metric/depth /da3_metric/camera_info \
      /da3_metric/scale -o tag_check

Run:
  python3 validate_apriltag.py tag_check --tag-size 0.025 --tag-spacing 0.3 --cols 6 --rows 6

Grid args match the Kalibr target yaml (tagCols, tagRows, tagSize [m], tagSpacing [ratio]).

Reports
  * per-frame scale error  = median(metric / true) over the board  (ideal 1.0)
  * per-pixel AbsRel, RMSE, delta<1.05/1.25, binned by board distance
  * PnP reprojection error (detection quality)
Outputs: printed summary, CSV, PNG plots (if matplotlib), and a few overlay images.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('needs OpenCV with the aruco module (python3-opencv)')


# ----------------------------------------------------------------------------
# AprilTag detection (OpenCV 4.6 .. 5.x)
# ----------------------------------------------------------------------------

class TagDetector:
    def __init__(self, border_bits, upscale=2.0):
        self.upscale = float(upscale)
        ar = cv2.aruco
        self.dict = ar.getPredefinedDictionary(ar.DICT_APRILTAG_36h11)
        params = ar.DetectorParameters() if hasattr(ar, 'DetectorParameters') \
            else ar.DetectorParameters_create()
        params.markerBorderBits = int(border_bits)
        params.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 3
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.minMarkerPerimeterRate = 0.01      # small tags
        self.params = params
        self.detector = ar.ArucoDetector(self.dict, params) \
            if hasattr(ar, 'ArucoDetector') else None

    def detect(self, gray):
        # Small tags (2.5 cm is ~14 px at 1 m): detect on an upscaled image
        g = gray if self.upscale == 1.0 else cv2.resize(
            gray, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(g)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(g, self.dict, parameters=self.params)
        if ids is None:
            return {}
        # pixel-centre-correct mapping back to the original resolution
        return {int(i): (c.reshape(4, 2).astype(np.float64) + 0.5) / self.upscale - 0.5
                for i, c in zip(ids.ravel(), corners)}


def quad_center(c):
    """Projective centre of a quad = intersection of its diagonals."""
    p1, p3 = np.append(c[0], 1), np.append(c[2], 1)
    p2, p4 = np.append(c[1], 1), np.append(c[3], 1)
    x = np.cross(np.cross(p1, p3), np.cross(p2, p4))
    return x[:2] / x[2] if abs(x[2]) > 1e-12 else c.mean(0)


class Grid:
    """Kalibr aprilgrid: id = row * cols + col, pitch = size * (1 + spacing)."""

    def __init__(self, cols, rows, size, spacing):
        self.cols, self.rows, self.size = cols, rows, size
        self.pitch = size * (1.0 + spacing)

    def center(self, tag_id):
        if not 0 <= tag_id < self.cols * self.rows:
            return None
        r, c = divmod(tag_id, self.cols)
        return np.array([c * self.pitch + self.size / 2, r * self.pitch + self.size / 2, 0.0])


# ----------------------------------------------------------------------------
# Per-frame evaluation
# ----------------------------------------------------------------------------

def undistort_norm(uv, K, D):
    return cv2.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)


def evaluate_frame(gray, depth, K, D, grid, det, min_tags=8, max_reproj=1.5, erode_px=4):
    tags = det.detect(gray)
    ids = [i for i in tags if grid.center(i) is not None]
    if len(ids) < min_tags:
        return None, tags
    obj = np.array([grid.center(i) for i in ids])
    img = np.array([quad_center(tags[i]) for i in ids])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, tags
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    if reproj > max_reproj:
        return {'rejected': 'reproj', 'reproj': reproj, 'n_tags': len(ids)}, tags

    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]                                   # board normal in camera frame
    d = float(n @ tvec.ravel())                   # plane: n . X = d
    board_dist = float(np.linalg.norm(tvec))

    # Board region in the image: hull of all detected tag corners, eroded
    H, W = depth.shape
    sx, sy = W / gray.shape[1], H / gray.shape[0]
    pts = np.concatenate([tags[i] for i in ids]) * [sx, sy]
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 1)
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    vs, us = np.nonzero(mask)
    if len(us) < 50:
        return None, tags

    z_pred = depth[vs, us].astype(np.float64)
    valid = np.isfinite(z_pred) & (z_pred > 0)
    if valid.sum() < 50:
        return {'rejected': 'no_depth', 'reproj': reproj, 'n_tags': len(ids)}, tags
    # true z-depth of the plane along each pixel ray (pixels in depth-image coords)
    Kd = K.copy()
    Kd[0, :] *= sx
    Kd[1, :] *= sy
    nrm = undistort_norm(np.stack([us[valid], vs[valid]], 1).astype(np.float64), Kd, D)
    rays = np.concatenate([nrm, np.ones((len(nrm), 1))], 1)
    z_true = d / (rays @ n)
    z_pred = z_pred[valid]
    good = z_true > 0.05
    z_true, z_pred = z_true[good], z_pred[good]
    ratio = z_pred / z_true
    return {
        'n_tags': len(ids), 'reproj': reproj, 'dist': board_dist,
        'z_board': float(np.median(z_true)),
        'ratio_med': float(np.median(ratio)),
        'ratio_iqr': float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
        'absrel': float(np.mean(np.abs(ratio - 1))),
        'n_px': int(len(ratio)),
        'z_true': z_true, 'z_pred': z_pred,
        'tilt_deg': float(np.degrees(np.arccos(abs(n[2])))),
    }, tags


def metrics(z_true, z_pred):
    r = np.maximum(z_pred / z_true, z_true / z_pred)
    return {
        'n': len(z_true),
        'absrel': float(np.mean(np.abs(z_pred - z_true) / z_true)),
        'rmse': float(np.sqrt(np.mean((z_pred - z_true) ** 2))),
        'bias': float(np.median(z_pred / z_true) - 1),
        'd105': float(np.mean(r < 1.05)),
        'd125': float(np.mean(r < 1.25)),
    }


# ----------------------------------------------------------------------------
# Bag
# ----------------------------------------------------------------------------

def iter_bag(bag, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise RuntimeError(f'topics not in bag: {missing}; have: {sorted(types)}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    cls = {t: get_message(types[t]) for t in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, cls[topic])


def stamp_ns(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def to_gray(msg):
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    enc = msg.encoding.lower()
    if enc in ('mono8', '8uc1'):
        return buf.reshape(msg.height, msg.step)[:, :msg.width].copy()
    if enc in ('rgb8', 'bgr8'):
        im = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY if enc == 'rgb8' else cv2.COLOR_BGR2GRAY)
    if enc in ('rgba8', 'bgra8'):
        im = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY if enc == 'rgba8' else cv2.COLOR_BGRA2GRAY)
    if enc in ('yuv422_yuy2', 'yuyv', 'yuv422'):
        im = buf.reshape(msg.height, msg.step)[:, :msg.width * 2].reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(im, cv2.COLOR_YUV2GRAY_YUY2)
    raise ValueError(f'unsupported image encoding {msg.encoding}')


def to_depth(msg):
    if msg.encoding.upper() not in ('32FC1', '32FC'):
        raise ValueError(f'expected 32FC1 depth, got {msg.encoding}')
    a = np.frombuffer(bytes(msg.data), '>f4' if msg.is_bigendian else '<f4')
    return a.reshape(msg.height, msg.step // 4)[:, :msg.width]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--image-topic', default='/image_throttled')
    ap.add_argument('--depth-topic', default='/da3_metric/depth')
    ap.add_argument('--info-topic', default='/da3_metric/camera_info')
    ap.add_argument('--tag-size', type=float, default=0.025, help='tagSize [m]')
    ap.add_argument('--tag-spacing', type=float, default=0.3, help='tagSpacing (ratio)')
    ap.add_argument('--cols', type=int, default=6, help='tagCols')
    ap.add_argument('--rows', type=int, default=6, help='tagRows')
    ap.add_argument('--border-bits', type=int, default=0,
                    help='tag black border width in bits (Kalibr grids: 2). 0 = auto')
    ap.add_argument('--upscale', type=float, default=2.0,
                    help='upscale image before detection (helps small tags; 1 = off)')
    ap.add_argument('--min-tags', type=int, default=8)
    ap.add_argument('--max-reproj', type=float, default=1.5, help='PnP RMS px')
    ap.add_argument('--bins', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument('--out', default=None)
    ap.add_argument('--overlays', type=int, default=6, help='overlay images to save')
    args = ap.parse_args()

    bag = args.bag.rstrip('/')
    out = args.out or bag + '_tags'
    os.makedirs(out, exist_ok=True)
    grid = Grid(args.cols, args.rows, args.tag_size, args.tag_spacing)

    # Pass 1: camera_info + raw images keyed by stamp (only those with a depth pair)
    info, images, depth_stamps = None, {}, set()
    for topic, msg in iter_bag(bag, [args.info_topic, args.depth_topic]):
        if topic == args.info_topic and info is None:
            info = msg
        elif topic == args.depth_topic:
            depth_stamps.add(stamp_ns(msg.header))
    if info is None:
        sys.exit('no camera_info in bag')
    for topic, msg in iter_bag(bag, [args.image_topic]):
        s = stamp_ns(msg.header)
        if s in depth_stamps:
            images[s] = to_gray(msg)
    print(f'bag: {bag}\n  depth frames: {len(depth_stamps)}  images with matching stamp: '
          f'{len(images)}')
    if not images:
        sys.exit('no image/depth pairs with identical stamps - check --image-topic')

    K = np.array(info.k, float).reshape(3, 3)
    D = np.array(info.d, float)
    # camera_info is at depth resolution; scale K to the raw image resolution
    any_img = next(iter(images.values()))
    Ki = K.copy()
    Ki[0, :] *= any_img.shape[1] / info.width
    Ki[1, :] *= any_img.shape[0] / info.height

    # Border bits: Kalibr aprilgrids use a 2-bit border, standard tags 1
    if args.border_bits == 0:
        sample = list(images.values())[::max(1, len(images) // 20)]
        counts = {b: sum(len(TagDetector(b, args.upscale).detect(g)) for g in sample)
                  for b in (2, 1)}
        border = max(counts, key=counts.get)
        print(f'  border bits auto: {border} (tags found in sample: {counts})')
    else:
        border = args.border_bits
    det = TagDetector(border, args.upscale)

    rows, all_t, all_p, all_d = [], [], [], []
    rejected = {}
    n_overlay = 0
    for s, msg in iter_bag(bag, [args.depth_topic]):
        s = stamp_ns(msg.header)
        if s not in images:
            continue
        depth = to_depth(msg)
        res, tags = evaluate_frame(images[s], depth, Ki, D, grid, det, args.min_tags,
                                   args.max_reproj)
        if res is None:
            continue
        if 'rejected' in res:
            rejected[res['rejected']] = rejected.get(res['rejected'], 0) + 1
            continue
        rows.append([s * 1e-9, res['dist'], res['z_board'], res['tilt_deg'], res['n_tags'],
                     res['reproj'], res['ratio_med'], res['ratio_iqr'], res['absrel'],
                     res['n_px']])
        all_t.append(res['z_true'])
        all_p.append(res['z_pred'])
        all_d.append(np.full(len(res['z_true']), res['z_board']))
        if n_overlay < args.overlays and len(rows) % max(1, 10) == 1:
            save_overlay(os.path.join(out, f'overlay_{n_overlay:02d}.png'), images[s], depth,
                         tags, res)
            n_overlay += 1

    print(f'  frames evaluated: {len(rows)}   rejected: {rejected or "none"}')
    if not rows:
        print('\nNo usable frames: board not detected (try --border-bits 1/2, check '
              '--cols/--rows), too few tags, or no depth on the board.')
        return
    A = np.array(rows)
    zt, zp, zb = np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_d)

    print('\n[Per-frame scale error] median(metric / true) over the board, ideal 1.000')
    r = A[:, 6]
    print(f'  n={len(r)} median={np.median(r):.4f} mean={r.mean():.4f} std={r.std():.4f} '
          f'p5={np.percentile(r, 5):.4f} p95={np.percentile(r, 95):.4f}')
    print(f'  frames within 5%: {np.mean(np.abs(r - 1) < 0.05) * 100:.1f}%   '
          f'within 10%: {np.mean(np.abs(r - 1) < 0.10) * 100:.1f}%')
    print(f'  within-board spread (IQR of ratio, DA3 shape error): median '
          f'{np.median(A[:, 7]):.4f}')
    print(f'  PnP reprojection: median {np.median(A[:, 5]):.2f} px, '
          f'tags/frame median {np.median(A[:, 4]):.0f}')

    print('\n[Per-pixel depth error on the board]')
    m = metrics(zt, zp)
    print(f'  all: n={m["n"]} AbsRel={m["absrel"]:.3f} RMSE={m["rmse"] * 100:.1f} cm '
          f'bias={m["bias"] * 100:+.1f}% d<1.05={m["d105"] * 100:.1f}% '
          f'd<1.25={m["d125"] * 100:.1f}%')
    for lo, hi in zip(args.bins[:-1], args.bins[1:]):
        sel = (zb >= lo) & (zb < hi)
        if sel.sum() < 100:
            continue
        mb = metrics(zt[sel], zp[sel])
        nf = int(np.sum((A[:, 2] >= lo) & (A[:, 2] < hi)))
        print(f'    {lo:.1f}-{hi:.1f} m: frames={nf:4d} AbsRel={mb["absrel"]:.3f} '
              f'RMSE={mb["rmse"] * 100:.1f} cm bias={mb["bias"] * 100:+.1f}% '
              f'd<1.05={mb["d105"] * 100:.1f}%')

    with open(os.path.join(out, 'tag_frames.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'board_dist_m', 'board_z_m', 'tilt_deg', 'n_tags', 'reproj_px',
                    'ratio_median', 'ratio_iqr', 'absrel', 'n_px'])
        w.writerows(A.tolist())
    plots(out, A, zt, zp)
    print(f'\nresults in {out}/')


def save_overlay(path, gray, depth, tags, res):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in tags.values():
        cv2.polylines(vis, [c.astype(np.int32)], True, (0, 255, 0), 1)
    d = depth.copy()
    d[~np.isfinite(d)] = 0
    hi = np.percentile(d[d > 0], 98) if np.any(d > 0) else 1.0
    dv = cv2.applyColorMap(np.clip(d / hi * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dv = cv2.resize(dv, (vis.shape[1], vis.shape[0]))
    txt = (f"z={res['z_board']:.2f}m  ratio={res['ratio_med']:.3f}  "
           f"tags={res['n_tags']}  reproj={res['reproj']:.2f}px")
    cv2.putText(vis, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(path, np.hstack([vis, dv]))


def plots(out, A, zt, zp):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not available - skipping plots)')
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(A[:, 2], A[:, 6], '.', ms=4)
    ax[0].axhline(1, color='k', lw=0.8)
    ax[0].set_xlabel('board depth [m]')
    ax[0].set_ylabel('median metric / true')
    ax[0].set_title('Per-frame scale error')
    ax[0].grid(alpha=0.3)
    ax[1].plot(A[:, 0] - A[0, 0], A[:, 6], '.-', ms=3, lw=0.6)
    ax[1].axhline(1, color='k', lw=0.8)
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('median metric / true')
    ax[1].set_title('Scale error over time')
    ax[1].grid(alpha=0.3)
    idx = np.random.default_rng(0).choice(len(zt), min(20000, len(zt)), replace=False)
    lim = [0, max(zt[idx].max(), zp[idx].max()) * 1.05]
    ax[2].plot(zt[idx], zp[idx], ',', alpha=0.3)
    ax[2].plot(lim, lim, 'k-', lw=0.8)
    ax[2].set_xlim(lim)
    ax[2].set_ylim(lim)
    ax[2].set_xlabel('true depth (AprilGrid) [m]')
    ax[2].set_ylabel('metric DA3 depth [m]')
    ax[2].set_title('Per-pixel depth')
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'apriltag_validation.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
