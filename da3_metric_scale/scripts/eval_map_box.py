#!/usr/bin/env python3
"""Map / depth accuracy against a box tracked by OptiTrack.

Put OptiTrack markers on a box, make it a rigid body in Motive (e.g. "box"), measure the box, and
record while flying (or carrying the drone) towards it at different distances and angles:

    ros2 bag record -o box_check --topics /esdf_grid_cloud /mavros/mavros/local_position/pose \\
        /vrpn_mocap/box/pose /vrpn_mocap/dilon/pose_throttled

    pip install rosbags mcap matplotlib numpy
    python3 eval_map_box.py box_check --size 0.57 0.57 0.45 --map-z-offset 0

The true box (OptiTrack pivot + size) is compared with the mapped obstacle surface
(/esdf_grid_cloud, ESDF <= --surface) in every map message, on the box face turned to the camera:

  1. Point error     each mapped point vs the true face along its own camera ray
                     (+ = mapped too far, - = too close).
  2. Distance ratio  per map message: mapped / true distance to the face (1.000 = perfect).
                     A constant ratio over distance = depth scale error.
  3. Sideways        where the mapped face lies across the true face; a shift = camera yaw /
                     extrinsic error rather than depth.

--pivot front (default): the markers are on the face the drone looks at. The box is oriented to
face the drone, --size is depth (away from the camera) width height, and it stands on the floor.
--pivot top|center|bottom: pivot there, box axes from the rigid body, --size along its x y z
(or --pivot-offset dx dy dz = box centre relative to the pivot).
Heights: the map is in the EKF frame; it is shifted by the measured EKF - OptiTrack offset of the
drone (needs /vrpn_mocap/dilon/...) or by --map-z-offset.
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ppb', _here / 'plot_planner_bag.py')
ppb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppb)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--size', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                    help='box size along the rigid-body x, y, z axes [m]')
    ap.add_argument('--pivot', choices=['front', 'top', 'center', 'bottom'], default='front',
                    help="front: markers on the face the camera looks at (box oriented to face the "
                         "drone, --size = depth width height, box standing on the floor); "
                         "top/center/bottom: pivot there, box axes from the rigid body")
    ap.add_argument('--pivot-offset', type=float, nargs=3, default=None,
                    help='box centre relative to the pivot, rigid-body axes [m] (overrides --pivot)')
    ap.add_argument('--box-topic', default='/vrpn_mocap/box/pose')
    ap.add_argument('--map-topic', default=None, help='default: /esdf_grid_cloud or esdf_cloud/grid')
    ap.add_argument('--map-z-offset', type=float, default=None,
                    help='map_mocap_z_offset used when flying [m] (default: measured)')
    ap.add_argument('--surface', type=float, default=0.10, help='mapped surface = ESDF <= this [m]')
    ap.add_argument('--lat-margin', type=float, default=0.3,
                    help='use mapped points up to this far beside the face [m]')
    ap.add_argument('--vert-margin', type=float, default=-0.03,
                    help='... and this far above / below it; negative = stay inside the face, so the box top seen from above is not counted [m]')
    ap.add_argument('--behind', type=float, default=2.0, help='... and up to this far behind it [m]')
    ap.add_argument('--front', type=float, default=0.5, help='... and up to this far in front of it [m]')
    ap.add_argument('--no-floor', dest='on_floor', action='store_false',
                    help='front mode: box centre height from the pivot instead of standing on the floor')
    ap.add_argument('--floor-cut', type=float, default=0.15, help='ignore mapped points below [m]')
    ap.add_argument('--hfov', type=float, default=60.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    size = np.array(args.size)
    half = size / 2
    if args.pivot_offset is not None:
        pivot_off = np.array(args.pivot_offset)
    else:
        pivot_off = {'top': np.array([0, 0, -half[2]]), 'center': np.zeros(3),
                     'bottom': np.array([0, 0, half[2]]), 'front': None}[args.pivot]

    # ---------------------------------------------------------------- read
    ts = ppb.get_typestore(ppb.Stores.ROS2_JAZZY)
    ts.register(ppb.get_types_from_msg(ppb.ESDF_GRID_MSG, 'da3_metric_scale/msg/EsdfGrid'))
    src = ppb.open_bag(args.bag, ts)
    topics = src.topics
    map_topic = args.map_topic or next((t for t in ppb.MAP_TOPICS if t in topics), None)
    drone_moc = next((t for t in ppb.MOCAP_TOPICS if t in topics), None)
    if map_topic is None or args.box_topic not in topics:
        sys.exit(f'need a map topic and {args.box_topic}; bag has: {sorted(topics)}')
    want = {map_topic, args.box_topic, ppb.EKF_TOPIC} | ({drone_moc} if drone_moc else set())
    want &= set(topics)
    clouds, box, ekf, moc = [], [], [], []
    for top, mt, t_ns, raw, deser in src.messages(want):
        t = (t_ns - src.start_ns) * 1e-9
        msg = deser(raw, mt)
        if top == map_topic:
            pts = ppb.grid_surface(msg, args.surface) if top.endswith('esdf_grid') else ppb.cloud_xyzd(msg)
            clouds.append((t, pts))
        else:
            p, q = msg.pose.position, msg.pose.orientation
            row = (t, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            {args.box_topic: box, ppb.EKF_TOPIC: ekf}.get(top, moc).append(row)
    src.close()
    box, ekf, moc = (np.array(a) if a else np.zeros((0, 8)) for a in (box, ekf, moc))
    if not len(box):
        sys.exit('no box poses')
    drone = ekf if len(ekf) else moc
    if not len(drone):
        sys.exit('no drone pose (local_position or vrpn)')

    if args.map_z_offset is not None:
        dz = args.map_z_offset
    elif len(ekf) > 1 and len(moc) > 1:
        dz = float(np.median(np.interp(moc[:, 0], ekf[:, 0], ekf[:, 3]) - moc[:, 3]))
    else:
        dz = 0.0
        print('  no drone OptiTrack pose in the bag: assuming map z offset 0 (use --map-z-offset)')
    print(f'bag {args.bag}: map {map_topic}, {len(clouds)} map messages, {len(box)} box poses; '
          f'map z offset {dz:+.3f} m')

    # ---------------------------------------------------------------- box pose in the map frame
    bp = np.median(box[:, 1:4], axis=0) + np.array([0, 0, dz])
    move = np.max(np.linalg.norm(box[:, 1:4] - np.median(box[:, 1:4], axis=0), axis=1))
    if args.pivot == 'front' and args.pivot_offset is None:
        # markers on the face the drone looks at: that face's normal points to the drone
        d_mean = np.median(drone[:, 1:3], axis=0) - bp[:2]
        n = np.array([d_mean[0], d_mean[1], 0.0])
        n /= np.linalg.norm(n)
        R = np.column_stack([n, np.cross([0, 0, 1.0], n), [0, 0, 1.0]])   # x = face normal
        c_box = bp - n * half[0]
        if args.on_floor:
            c_box[2] = dz + half[2]
        mode = 'front face (faces the drone), standing on the floor' if args.on_floor else 'front face'
    else:
        q = box[len(box) // 2, 4:8]
        R = quat_to_R(q / np.linalg.norm(q))
        c_box = bp + R @ pivot_off
        mode = f'pivot {args.pivot}, axes from the rigid body'
    print(f'  box: {mode}; centre (map frame) {c_box.round(3)}, size {size}, '
          f'yaw {math.degrees(math.atan2(R[1, 0], R[0, 0])):.1f} deg, moved {move * 100:.1f} cm')
    to_local = lambda P: (P - c_box) @ R          # noqa: E731

    # ---------------------------------------------------------------- per map message
    pt_err, msg_rows, lat_rows, timeline = [], [], [], []
    for t, pts in clouds:
        if len(pts) == 0 or t < drone[0, 0] or t > drone[-1, 0]:
            continue
        cam = np.array([np.interp(t, drone[:, 0], drone[:, k]) for k in (1, 2, 3)])
        pts = pts[(pts[:, 3] <= args.surface) & (pts[:, 2] - dz > args.floor_cut)]
        o = to_local(cam[None, :])[0]
        tl = [t, float(np.linalg.norm(o - np.array([half[0], 0, 0]) * (args.pivot == 'front'))), len(pts), 0]
        timeline.append(tl)
        if len(pts) == 0:
            continue
        # the face turned most towards the camera
        cand = [(a, sg) for a in range(3) for sg in (-1, 1)]
        a, sg = max(cand, key=lambda f: (f[1] * o[f[0]] - half[f[0]]) / (np.linalg.norm(o) + 1e-9))
        if sg * o[a] <= half[a]:
            continue                                  # camera inside the slab of that face
        l1, l2 = [k for k in range(3) if k != a]
        loc = to_local(pts[:, :3])
        off = sg * loc[:, a] - half[a]                 # + in front of the face, - behind it
        dvec = loc - o
        r = np.linalg.norm(dvec, axis=1)
        u = dvec / r[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_pl = (sg * half[a] - o[a]) / u[:, a]     # distance along each ray to the face plane
        hit = o[None, :] + u * t_pl[:, None]           # where each ray crosses the face plane
        # vertical axis of the face: tight margin, so the box top (seen from above) is not counted
        up = int(np.argmax(np.abs(R[2, :])))
        mg = {k: (args.vert_margin if k == up else args.lat_margin) for k in (l1, l2)}
        sel = ((off > -args.behind) & (off < args.front) & np.isfinite(t_pl) & (t_pl > 0) &
               (np.abs(hit[:, l1]) <= half[l1] + mg[l1]) &
               (np.abs(hit[:, l2]) <= half[l2] + mg[l2]))
        tl[3] = int(sel.sum())
        if sel.sum() < 5:
            continue
        e = r[sel] - t_pl[sel]
        pt_err.append(e)
        t_true = sg * half[a] - o[a]                   # perpendicular distance camera -> face
        # distance along the ray through the face centre = perpendicular distance / cos
        fc = np.zeros(3)
        fc[a] = sg * half[a]
        t_c = np.linalg.norm(fc - o)
        msg_rows.append((t, t_c, t_c + np.median(e), np.median(r[sel] / t_pl[sel]), t_true))
        lat = l2 if l1 == up else l1                  # horizontal axis across the face
        # sign of the across-face axis as seen by the viewer: + = to the viewer's right
        fwd = -(R[:, a] * sg)
        right = np.cross(fwd, [0, 0, 1.0])
        sgn = 1.0 if np.dot(R[:, lat], right) >= 0 else -1.0
        lo_, hi_ = np.percentile(sgn * hit[sel, lat], [5, 95])
        lat_rows.append((t, sgn * np.median(hit[sel, lat]), lo_, hi_, lat, t_c))

    TL = np.array(timeline) if timeline else np.zeros((0, 4))
    if len(TL):
        print('\n  timeline (1 s bins): drone -> box face distance, map points kept, of them on the face')
        for t0 in np.arange(np.floor(TL[0, 0]), TL[-1, 0] + 1e-9, 1.0):
            m = (TL[:, 0] >= t0) & (TL[:, 0] < t0 + 1)
            if m.any():
                print(f'    {t0:5.0f} s  {np.median(TL[m, 1]):5.2f} m  {int(np.median(TL[m, 2])):6d}  '
                      f'{int(np.median(TL[m, 3])):6d}')
    if not pt_err:
        sys.exit('no mapped points in front of / behind the box face - check --size / --pivot, '
                 'or increase --behind / --lat-margin')
    allp_err = np.concatenate(pt_err)
    M = np.array(msg_rows)
    L = np.array(lat_rows)
    print(f'\n[1] per-point depth error vs the box face ({len(allp_err)} points, {len(M)} map messages)')
    print('    error = camera->mapped point - camera->true face, on the same camera ray')
    print(f'    median {np.median(allp_err):+.3f} m   mean {np.mean(allp_err):+.3f} m   '
          f'5-95 % {np.percentile(allp_err, 5):+.3f} .. {np.percentile(allp_err, 95):+.3f} m   (+ = mapped too far)')
    print('\n[2] per map message (median over the face)')
    print(f'    mapped / true distance: median {np.median(M[:, 3]):.3f}   (1.000 = perfect, >1 = too far)')
    bins = [0, 0.75, 1.25, 1.75, 2.5, 4.0, 10]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (M[:, 1] >= lo) & (M[:, 1] < hi)
        if m.sum() >= 3:
            print(f'      true {lo:4.2f}-{hi:4.2f} m: n {m.sum():4d}  ratio {np.median(M[m, 3]):.3f}  '
                  f'error {np.median(M[m, 2] - M[m, 1]):+.3f} m')
    if len(L):
        w = half[int(L[0, 4])]
        print('\n[3] sideways: where the mapped face lies across the true face (after projecting each '
              'point back onto the face plane)')
        ang = np.degrees(np.arctan2(L[:, 1], L[:, 5]))
        print(f'    centre offset {np.median(L[:, 1]):+.3f} m (+ = to the right as seen from the drone; '
              f'true face spans +-{w:.3f} m); mapped extent {np.median(L[:, 2]):+.3f} .. '
              f'{np.median(L[:, 3]):+.3f} m')
        print(f'    = bearing error {np.median(ang):+.1f} deg (+ = obstacles mapped to the right of '
              f'where they are): a camera / body yaw error (extrinsics or rigid-body heading), not depth')
    # where the box is relative to the drone heading (EKF yaw = OptiTrack rigid-body heading)
    if len(ekf):
        yaw = np.arctan2(2 * (ekf[:, 7] * ekf[:, 6] + ekf[:, 4] * ekf[:, 5]),
                         1 - 2 * (ekf[:, 5] ** 2 + ekf[:, 6] ** 2))
        fcw = c_box + R[:, 0] * half[0] if args.pivot == 'front' else c_box
        brg = np.arctan2(fcw[1] - ekf[:, 2], fcw[0] - ekf[:, 1])
        rel = np.degrees(np.arctan2(np.sin(brg - yaw), np.cos(brg - yaw)))
        print(f'\n    box face centre relative to the drone heading: median {np.median(rel):+.1f} deg '
              f'(+ = to the left). With the box centred in the camera image this is the '
              f'camera-vs-heading yaw offset.')

    # ---------------------------------------------------------------- plots
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    ax = axs[0, 0]
    allp = np.vstack([p for _, p in clouds if len(p)])
    allp = allp[(allp[:, 3] <= args.surface) & (allp[:, 2] - dz > args.floor_cut)]
    near = np.linalg.norm(allp[:, :2] - c_box[:2], axis=1) < 3.0
    ax.scatter(allp[near, 0], allp[near, 1], s=2, c='0.6', label='mapped surface (all messages)')
    corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]) * np.array([half[0], half[1], 0])
    cw = corners @ R.T + c_box
    ax.plot(cw[:, 0], cw[:, 1], 'r-', lw=2.5, label='true box')
    ax.plot(bp[0], bp[1], 'r+', ms=14, mew=2, label='OptiTrack pivot')
    ax.plot(drone[:, 1], drone[:, 2], color='tab:blue', lw=1, label='drone')
    ax.set_aspect('equal')
    ax.set_xlim(c_box[0] - 3, c_box[0] + 3)
    ax.set_ylim(c_box[1] - 3, c_box[1] + 3)
    ax.set_title('Top view: true box vs mapped surface')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axs[0, 1]
    ax.hist(np.clip(allp_err, -0.5, 1.5), bins=80, color='tab:gray')
    ax.axvline(0, color='r')
    ax.set_xlabel('mapped - true distance along the camera ray [m]  (+ = too far)')
    ax.set_title(f'[1] Per-point depth error (median {np.median(allp_err):+.3f} m)')
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    ax.scatter(M[:, 1], M[:, 2], s=10, c=M[:, 0], cmap='viridis')
    lim = max(M[:, 1].max(), M[:, 2].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', label='perfect')
    ax.plot([0, lim], [0, lim * np.median(M[:, 3])], 'k:', label=f'ratio {np.median(M[:, 3]):.2f}')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('true distance camera -> face centre [m]')
    ax.set_ylabel('mapped distance [m]')
    ax.set_title('[2] Mapped vs true distance per map message (colour = time)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axs[1, 1]
    if len(L):
        w = half[int(L[0, 4])]
        ax.fill_between(L[:, 0], L[:, 2], L[:, 3], color='tab:gray', alpha=0.4, label='mapped extent (5-95 %)')
        ax.plot(L[:, 0], L[:, 1], color='k', lw=1, label='mapped centre (+ = right, seen from drone)')
        ax.axhline(w, color='r')
        ax.axhline(-w, color='r', label='true face edges')
        ax.legend(fontsize=8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('across the face [m]')
    ax.set_title('[3] Sideways position of the mapped face')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = args.out or f'{str(Path(args.bag)).rstrip("/")}_box_eval.png'
    fig.savefig(out, dpi=180)
    print(f'\nsaved {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
