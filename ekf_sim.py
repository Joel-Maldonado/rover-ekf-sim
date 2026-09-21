"""A rover in 2D with an IMU (gyro + forward accelerometer, 100 Hz) and a stereo
camera that sees known markers (range + bearing, 10 Hz). Compares dead
reckoning, camera-only fixes, and an EKF with and without a gyro-bias state.

Usage: python ekf_sim.py [outdir]    (writes plots + prints a results table)
"""
import sys

import numpy as np

DT = 0.01               # IMU period, s
CAM_EVERY = 10          # camera frame every 10 IMU samples: 10 Hz
T_END = 180.0           # s

GYRO_SIGMA = 0.005      # rad/s per sample
GYRO_BIAS = 0.02        # rad/s, about 1.1 deg/s: an uncalibrated MEMS gyro
ACC_SIGMA = 0.05        # m/s^2 per sample
ACC_BIAS = 0.02         # m/s^2
BEARING_SIGMA = 0.02    # rad
CAM_RANGE = 15.0        # m, beyond this the stereo depth is useless
CAM_FOV = np.radians(55)  # half-angle, about a ZED 2's horizontal view


def range_sigma(r):
    # Stereo depth error grows with the square of distance.
    return 0.05 + 0.01 * r * r


# Eight markers around a loop of radius 12 m centered on the origin, some
# inside the loop and some outside, with a gap on the left side.
_ang = np.radians([0, 40, 80, 120, 200, 245, 290, 330])
_rad = np.array([15, 9, 15, 9, 15, 9, 15, 9])
LANDMARKS = np.c_[_rad * np.cos(_ang), _rad * np.sin(_ang)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate(rng):
    """Ground truth plus raw sensor readings."""
    n = int(T_END / DT)
    t = np.arange(n) * DT
    v = 1.0 + 0.3 * np.sin(0.1 * t)                 # m/s
    w = v * (1 / 12 + 0.03 * np.sin(0.3 * t))       # laps of a wobbly 12 m loop
    a = np.gradient(v, DT)
    truth = np.zeros((n, 4))                         # x, y, theta, v
    truth[0] = [0.0, -12.0, 0.0, v[0]]
    for k in range(1, n):
        x, y, th, _ = truth[k - 1]
        truth[k] = [x + v[k - 1] * np.cos(th) * DT, y + v[k - 1] * np.sin(th) * DT, th + w[k - 1] * DT, v[k]]
    gyro = w + GYRO_BIAS + rng.normal(0, GYRO_SIGMA, n)
    acc = a + ACC_BIAS + rng.normal(0, ACC_SIGMA, n)
    frames = {}
    for k in range(0, n, CAM_EVERY):
        x, y, th, _ = truth[k]
        seen = []
        for j, (lx, ly) in enumerate(LANDMARKS):
            r = np.hypot(lx - x, ly - y)
            b = wrap(np.arctan2(ly - y, lx - x) - th)
            if r < CAM_RANGE and abs(b) < CAM_FOV:
                seen.append((j, r + rng.normal(0, range_sigma(r)), wrap(b + rng.normal(0, BEARING_SIGMA))))
        frames[k] = seen
    return t, truth, gyro, acc, frames


def dead_reckon(truth, gyro, acc):
    est = np.zeros_like(truth)
    est[0] = truth[0]
    for k in range(1, len(truth)):
        x, y, th, v = est[k - 1]
        est[k] = [x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, th + gyro[k - 1] * DT, v + acc[k - 1] * DT]
    return est


def camera_only(truth, frames):
    """Gauss-Newton fix of (x, y, theta) from each frame that sees 2+ markers."""
    n = len(truth)
    est = np.full((n, 3), np.nan)
    fixes = 0
    for k, seen in frames.items():
        if len(seen) < 2:
            continue
        # Closed-form start: rotate the markers as the camera sees them onto
        # where they are on the map (2D Kabsch), then polish with Gauss-Newton.
        cam = np.array([[r * np.cos(b), r * np.sin(b)] for _, r, b in seen])
        world = LANDMARKS[[j for j, _, _ in seen]]
        c0, w0 = cam - cam.mean(0), world - world.mean(0)
        th = np.arctan2(np.sum(c0[:, 0] * w0[:, 1] - c0[:, 1] * w0[:, 0]), np.sum(c0 * w0))
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        g = np.array([*(world.mean(0) - rot @ cam.mean(0)), th])
        for _ in range(8):
            res, J = [], []
            for j, r, b in seen:
                dx, dy = LANDMARKS[j] - g[:2]
                q = dx * dx + dy * dy
                res += [r - np.sqrt(q), wrap(b - (np.arctan2(dy, dx) - g[2]))]
                J += [[-dx / np.sqrt(q), -dy / np.sqrt(q), 0], [dy / q, -dx / q, -1]]
            step = np.linalg.lstsq(np.array(J), np.array(res), rcond=None)[0]
            g += step
        if np.linalg.norm(step) > 1e-3:
            continue                                 # didn't converge: no fix this frame
        est[k] = g
        fixes += 1
    # Hold the last fix between frames, like you would on the robot.
    last = truth[0, :3]
    for k in range(n):
        if np.isnan(est[k, 0]):
            est[k] = last
        else:
            last = est[k]
    return est, fixes


def ekf(truth, gyro, acc, frames, estimate_bias=True, wrap_bearing=True):
    s = np.array([*truth[0], 0.0])                   # x, y, theta, v, gyro bias
    P = np.diag([0.01, 0.01, 0.001, 0.01, 0.05 ** 2 if estimate_bias else 0.0])
    q_bias = (1e-4) ** 2 * DT if estimate_bias else 0.0
    # The velocity term leaves room for the accelerometer bias, which isn't a state.
    Q = np.diag([1e-6, 1e-6, (GYRO_SIGMA * DT) ** 2, (ACC_SIGMA * DT) ** 2 + ACC_BIAS ** 2 * DT, q_bias])
    est = np.zeros((len(truth), 5))
    sig = np.zeros((len(truth), 2))
    for k in range(len(truth)):
        if k in frames:
            for j, r, b in frames[k]:
                dx, dy = LANDMARKS[j] - s[:2]
                q = dx * dx + dy * dy
                rh = np.sqrt(q)
                h = np.array([rh, np.arctan2(dy, dx) - s[2]])
                H = np.array([[-dx / rh, -dy / rh, 0, 0, 0], [dy / q, -dx / q, -1, 0, 0]])
                R = np.diag([range_sigma(rh) ** 2, BEARING_SIGMA ** 2])
                innov = np.array([r, b]) - h
                if wrap_bearing:
                    innov[1] = wrap(innov[1])
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                s = s + K @ innov
                I_KH = np.eye(5) - K @ H
                P = I_KH @ P @ I_KH.T + K @ R @ K.T  # Joseph form: P stays symmetric
        est[k] = s
        sig[k] = np.sqrt([P[0, 0], P[1, 1]])
        x, y, th, v, bg = s
        F = np.eye(5)
        F[0, 2], F[0, 3] = -v * np.sin(th) * DT, np.cos(th) * DT
        F[1, 2], F[1, 3] = v * np.cos(th) * DT, np.sin(th) * DT
        F[2, 4] = -DT
        s = np.array([x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, th + (gyro[k] - bg) * DT, v + acc[k] * DT, bg])
        P = F @ P @ F.T + Q
    return est, sig


def pos_err(truth, est):
    return np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    seeds = range(20)
    rows = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        t, truth, gyro, acc, frames = simulate(rng)
        dr = dead_reckon(truth, gyro, acc)
        cam, fixes = camera_only(truth, frames)
        runs = {
            "dead reckoning": dr,
            "camera only": cam,
            "EKF, no bias state": ekf(truth, gyro, acc, frames, estimate_bias=False)[0],
            "EKF": ekf(truth, gyro, acc, frames)[0],
            "EKF, bearing not wrapped": ekf(truth, gyro, acc, frames, wrap_bearing=False)[0],
        }
        for name, est in runs.items():
            e = pos_err(truth, est)
            rows.setdefault(name, []).append((np.median(e), np.sqrt(np.mean(e ** 2)), e.max(), e[-1]))
        rows.setdefault("_fixes", []).append(fixes / len(frames))
        if seed == 0:
            keep = (t, truth, runs, frames, ekf(truth, gyro, acc, frames))
    print(f"{len(seeds)} runs of {T_END:.0f} s; camera fixes in {100 * np.mean(rows.pop('_fixes')):.0f}% of frames")
    counts = np.bincount([len(v) for v in keep[3].values()])
    print("seed 0: markers seen per frame:", {i: int(c) for i, c in enumerate(counts)})
    print(f"{'':26s} {'median':>9s} {'RMS':>9s} {'max':>9s} {'final':>9s}   (m, median over runs)")
    for name, vals in rows.items():
        v = np.median(np.array(vals), axis=0)
        print(f"{name:26s} {v[0]:9.2f} {v[1]:9.2f} {v[2]:9.2f} {v[3]:9.2f}")
    t, truth, runs, frames, (est, sig) = keep
    print(f"seed 0: final gyro bias estimate {est[-1, 4]:.4f} rad/s (true {GYRO_BIAS})")
    e = pos_err(truth, est)
    inside = np.mean(np.abs(est[:, 0] - truth[:, 0]) < 2 * sig[:, 0])
    print(f"seed 0: EKF x error inside its own 2-sigma {100 * inside:.0f}% of the time")
    for name in runs:
        ee = pos_err(truth, runs[name])
        print(f"seed 0: {name:26s} error at 30 s {ee[int(30 / DT)]:.2f} m, at 180 s {ee[-1]:.2f} m")
    plot(outdir, t, truth, runs)


def plot(outdir, t, truth, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    colors = {"truth": "#e8e8e8", "dead reckoning": "#e06c5a", "camera only": "#6aa9e0",
              "EKF": "#7fd17f", "EKF, no bias state": "#e0c36a", "EKF, bearing not wrapped": "#c792ea"}

    fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
    ax.plot(truth[:, 0], truth[:, 1], color=colors["truth"], lw=2.5, label="truth")
    for name in ["dead reckoning", "EKF"]:
        ax.plot(runs[name][:, 0], runs[name][:, 1], color=colors[name], lw=1.4, label=name)
    ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker="s", color="#aaaaaa", s=30, label="markers", zorder=3)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/paths.png", facecolor="#000000")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    for name in ["dead reckoning", "camera only", "EKF, no bias state", "EKF"]:
        ax.plot(t, pos_err(truth, runs[name]), color=colors[name], lw=1.2, label=name)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 200)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/errors.png", facecolor="#000000")


if __name__ == "__main__":
    main()
