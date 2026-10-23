#!/usr/bin/env python3
"""Estimate rigid-body -> IMU rotation (R_body_imu) and the mocap <-> IMU time offset
from a rosbag of mocap poses and FC gyro data.

Recording (props OFF, ~60 s):
  ros2 bag record /vrpn_mocap/dilon/pose /mavros/mavros/imu/data_raw -o body_imu_calib
  Hold the drone and rotate it by hand about ALL three axes (roll, pitch, yaw), with
  varied speed, a few times each. Keep it inside the mocap volume, markers visible.
  Use the RAW (unthrottled) mocap topic if possible: higher rate = better.

Run:
  python3 calib_body_imu.py body_imu_calib --kalibr-timeshift 0.085

Method
  * mocap angular velocity in the body frame from consecutive orientations
  * resample mocap and gyro onto a common grid, low-pass
  * time offset from cross-correlation of |omega| (independent of the rotation)
  * R_imu_body by least squares (Kabsch) on omega_imu = R * omega_body + bias

Output: R_body_imu for da3_metric_scale.yaml (same matrix as R_body_flu in
mocap_to_mavros.yaml, if the MAVROS IMU frame is FLU base_link), gyro bias, residuals,
and a suggested mocap_time_offset.
"""
import argparse
import math
import sys

import numpy as np


# ----------------------------------------------------------------------------
# math
# ----------------------------------------------------------------------------

def quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * w
    return th / (2 * math.sin(th)) * w


def body_rates(t, q):
    """Body-frame angular velocity from orientations (world_R_body)."""
    Rs = [quat_to_rot(qi) for qi in q]
    tm, om = [], []
    for k in range(len(Rs) - 1):
        dt = t[k + 1] - t[k]
        if dt <= 1e-4 or dt > 0.1:
            continue
        om.append(rot_log(Rs[k].T @ Rs[k + 1]) / dt)
        tm.append(0.5 * (t[k] + t[k + 1]))
    return np.array(tm), np.array(om)


def smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.stack([np.convolve(x[:, i], k, mode='same') for i in range(x.shape[1])], 1)


def resample(t_src, x_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, x_src[:, i]) for i in range(x_src.shape[1])], 1)


def estimate_offset(tm, wm, ti, wi, fs=200.0, max_lag=0.5, smooth_s=0.03):
    """Returns d such that mocap(t + d) matches imu(t), i.e. d = t_mocap - t_imu
    for the same physical instant."""
    t0 = max(tm[0], ti[0]) + max_lag
    t1 = min(tm[-1], ti[-1]) - max_lag
    if t1 - t0 < 5.0:
        raise RuntimeError('less than 5 s of overlapping mocap + IMU data')
    tg = np.arange(t0, t1, 1.0 / fs)
    n = max(1, int(smooth_s * fs))
    a = np.linalg.norm(smooth(resample(ti, wi, tg), n), axis=1)
    a = (a - a.mean()) / (a.std() + 1e-12)
    lags = np.arange(-int(max_lag * fs), int(max_lag * fs) + 1)
    corr = []
    for L in lags:
        b = np.linalg.norm(smooth(resample(tm, wm, tg + L / fs), n), axis=1)
        b = (b - b.mean()) / (b.std() + 1e-12)
        corr.append(np.mean(a * b))
    corr = np.array(corr)
    k = int(np.argmax(corr))
    d = lags[k] / fs
    if 0 < k < len(corr) - 1:                       # parabolic refinement
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            d += 0.5 * (y0 - y2) / den / fs
    return d, float(corr[k])


def estimate_rotation(wb, wi):
    """omega_imu = R_ib * omega_body + b  ->  (R_ib, b)."""
    mb, mi = wb.mean(0), wi.mean(0)
    H = (wb - mb).T @ (wi - mi)
    U, S, Vt = np.linalg.svd(H)
    Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dm @ U.T
    b = mi - R @ mb
    return R, b


def nearest_axis_aligned(R):
    """Closest rotation whose entries are in {-1, 0, 1}."""
    A = np.zeros((3, 3))
    used = set()
    for i in np.argsort(-np.abs(R).max(axis=1)):
        order = np.argsort(-np.abs(R[i]))
        for j in order:
            if j not in used:
                A[i, j] = np.sign(R[i, j])
                used.add(j)
                break
    return A


def angle_between(Ra, Rb):
    return math.degrees(np.linalg.norm(rot_log(Ra.T @ Rb)))


# ----------------------------------------------------------------------------
# bag
# ----------------------------------------------------------------------------

def read_bag(bag, pose_topic, imu_topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for tp in (pose_topic, imu_topic):
        if tp not in types:
            raise RuntimeError(f'topic {tp} not in bag; have: {sorted(types)}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[pose_topic, imu_topic]))
    cls = {tp: get_message(types[tp]) for tp in (pose_topic, imu_topic)}
    tp_, qp, ti, wi = [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        m = deserialize_message(data, cls[topic])
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if topic == pose_topic:
            o = m.pose.orientation
            tp_.append(t)
            qp.append([o.x, o.y, o.z, o.w])
        else:
            a = m.angular_velocity
            ti.append(t)
            wi.append([a.x, a.y, a.z])
    o1, o2 = np.argsort(tp_), np.argsort(ti)
    return (np.array(tp_)[o1], np.array(qp)[o1], np.array(ti)[o2], np.array(wi)[o2])


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def fmt(R):
    rows = [', '.join(f'{v: .4f}' for v in R[i]) for i in range(3)]
    return '[' + ',\n  '.join(rows) + ']'


def run(t_pose, q_pose, t_imu, w_imu, kalibr_shift=None, fs=200.0):
    tm, wm = body_rates(t_pose, q_pose)
    rate_pose = 1.0 / np.median(np.diff(t_pose))
    rate_imu = 1.0 / np.median(np.diff(t_imu))
    print(f'mocap: {len(t_pose)} poses @ ~{rate_pose:.0f} Hz   '
          f'IMU: {len(t_imu)} samples @ ~{rate_imu:.0f} Hz')

    d, peak = estimate_offset(tm, wm, t_imu, w_imu, fs=fs)
    print(f'\nTime offset  t_mocap - t_imu = {d * 1000:+.1f} ms   (correlation peak {peak:.3f})')
    if peak < 0.8:
        print('  WARNING: weak correlation - rotate more vigorously / longer, check topics')

    # Align, low-pass both to a common band (mocap rate limits the bandwidth)
    t0 = max(tm[0] - d, t_imu[0]) + 0.1
    t1 = min(tm[-1] - d, t_imu[-1]) - 0.1
    tg = np.arange(t0, t1, 1.0 / fs)
    n = max(1, int(fs / rate_pose))
    wb = smooth(resample(tm - d, wm, tg), n)
    wi = smooth(resample(t_imu, w_imu, tg), n)
    keep = (np.linalg.norm(wb, axis=1) > 0.2) & (np.linalg.norm(wi, axis=1) > 0.2)
    keep &= np.r_[False, np.ones(len(tg) - 2, bool), False]
    wb, wi = wb[keep], wi[keep]
    if len(wb) < 200:
        raise RuntimeError('not enough rotating samples (> 0.2 rad/s); rotate the drone more')

    sv = np.linalg.svd(wb - wb.mean(0), compute_uv=False) / math.sqrt(len(wb))
    print(f'Excitation (rad/s RMS per principal axis): {sv[0]:.2f} {sv[1]:.2f} {sv[2]:.2f}')
    if sv[2] < 0.3 * sv[0]:
        print('  WARNING: one axis barely excited - rotate about roll, pitch AND yaw')

    R_ib, b = estimate_rotation(wb, wi)
    res = wi - (wb @ R_ib.T + b)
    rms = np.sqrt(np.mean(np.sum(res ** 2, axis=1)))
    rel = rms / np.sqrt(np.mean(np.sum(wi ** 2, axis=1)))
    R_bi = R_ib.T
    A = nearest_axis_aligned(R_bi)

    print(f'\nFit residual: {rms:.3f} rad/s RMS ({rel * 100:.1f}% of signal)')
    print(f'Gyro bias (IMU frame): {b[0]:+.4f} {b[1]:+.4f} {b[2]:+.4f} rad/s')
    print('\nR_body_imu (columns = IMU axes in rigid-body frame):')
    print('  ' + fmt(R_bi))
    print('\nNearest axis-aligned rotation:')
    print('  ' + fmt(A))
    print(f'  estimated mount misalignment from that: {angle_between(A, R_bi):.2f} deg')

    print('\n--- paste into config/da3_metric_scale.yaml ---')
    print('    R_body_imu: [' + ',\n                 '.join(
        ', '.join(f'{v:.5f}' for v in R_bi[i]) for i in range(3)) + ']')
    print('--- and the same matrix as R_body_flu in config/mocap_to_mavros.yaml ---')

    if kalibr_shift is not None:
        mto = kalibr_shift + d
        print('\nTiming chain (Kalibr: t_imu = t_cam + timeshift_cam_imu):')
        print(f'  timeshift_cam_imu = {kalibr_shift * 1000:+.1f} ms, '
              f't_mocap - t_imu = {d * 1000:+.1f} ms')
        print(f'  -> mocap_time_offset: {mto:.4f}   (added to image stamp for pose lookup)')
    return R_bi, b, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--pose-topic', default='/vrpn_mocap/dilon/pose')
    ap.add_argument('--imu-topic', default='/mavros/mavros/imu/data_raw')
    ap.add_argument('--kalibr-timeshift', type=float, default=None,
                    help='timeshift_cam_imu from Kalibr (s) to compute mocap_time_offset')
    args = ap.parse_args()
    t_pose, q_pose, t_imu, w_imu = read_bag(args.bag.rstrip('/'), args.pose_topic,
                                            args.imu_topic)
    run(t_pose, q_pose, t_imu, w_imu, args.kalibr_timeshift)


if __name__ == '__main__':
    sys.exit(main())
