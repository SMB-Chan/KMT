"""Simple 3D visualisation of the flying-boat drone over an irregular sea.

Uses mpl_toolkits.mplot3d for lightweight, dependency-free rendering.

Two outputs are produced:
    results/scene_static.png   - 3D scene at a chosen moment
    results/scene_animated.gif - side-view animation (Pillow-based)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)


# ---------------------------------------------------------------------
def _wing_polygon(span: float, chord: float, alpha: float = 0.0):
    """Return vertices of a flat wing as an Nx3 array, centred at origin.

    Span is along the local y-axis (left-right), chord along x, lift on z.
    `alpha` (radians) is the pitch angle of the wing around the y-axis.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    # Four corners (x, y, z) in the body frame
    raw = np.array([
        [-chord / 2, -span / 2, 0.0],
        [ chord / 2, -span / 2, 0.0],
        [ chord / 2,  span / 2, 0.0],
        [-chord / 2,  span / 2, 0.0],
    ])
    # Apply pitch rotation around the y-axis
    Rz = np.array([[ca, 0, sa],
                   [ 0, 1,  0],
                   [-sa, 0, ca]])
    return raw @ Rz.T


def _hull_polygon(Lwl: float, Bwl: float):
    """Rectangular hull footprint, with z slightly raised."""
    return np.array([
        [-Lwl / 2, -Bwl / 2, -0.05],
        [ Lwl / 2, -Bwl / 2, -0.05],
        [ Lwl / 2,  Bwl / 2, -0.05],
        [-Lwl / 2,  Bwl / 2, -0.05],
    ])


def attitude_matrix(pitch, bank=0.0, heading=0.0):
    """Body x-forward/y-right/z-up to world north/east/up."""
    forward = np.array([math.cos(pitch) * math.cos(heading),
                        math.cos(pitch) * math.sin(heading), math.sin(pitch)])
    right = np.array([-math.sin(heading), math.cos(heading), 0.0])
    up = np.cross(forward, right)
    return np.column_stack((forward, right * math.cos(bank) - up * math.sin(bank),
                            up * math.cos(bank) + right * math.sin(bank)))


# ---------------------------------------------------------------------
def render_scene(traj, sea, *, time_index: int = -1,
                 output: str = "results/scene_static.png",
                 window_x: tuple[float, float] | None = None,
                 window_z: tuple[float, float] | None = None):
    """Render a single 3D frame with a side panel of control-input logs.

    traj : list of dicts, each with at least 'x', 'z', 'Vx', 'Vz',
           'alpha', 'throttle', 'eta', 'cmd_alt', 'cmd_speed',
           'mode' (string: 'TAKEOFF'/'LAND'/'WAYPOINT'/'STANDBY')
    sea  : Ocean / RealOcean
    """
    # State at the chosen moment
    if not traj:
        raise ValueError("trajectory must not be empty")
    time_index = time_index % len(traj)
    info = traj[time_index]
    x_a = info["x"]; z_a = info["z"]; alpha = info["alpha"]
    y_a = info.get("y", 0.0)
    rotation = attitude_matrix(alpha, info.get("bank", 0), info.get("heading", 0))
    offset = np.array([x_a, y_a, z_a])

    # Window:  centred on the aircraft
    if window_x is None:
        window_x = (x_a - 100.0, x_a + 100.0)
    if window_z is None:
        window_z = (-8.0, max(35.0, z_a + 12.0))

    # Build a wider wave surface mesh on the window
    xs = np.linspace(window_x[0], window_x[1], 161)
    ys = np.linspace(y_a - 30.0, y_a + 30.0, 31)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    from ocean_directional import DirectionalOcean, sea_eta_1d
    if isinstance(sea, DirectionalOcean):
        Z = sea.eta(xs, ys, info["t"])
    else:
        Z = np.tile(sea_eta_1d(sea, xs, info["t"]), (Y.shape[0], 1))

    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(5, 2, width_ratios=[2.4, 1.0], height_ratios=[3, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:4, 0], projection="3d")
    ax_thr = fig.add_subplot(gs[0, 1])
    ax_pit = fig.add_subplot(gs[1, 1], sharex=ax_thr)
    ax_alt = fig.add_subplot(gs[2, 1], sharex=ax_thr)
    ax_spd = fig.add_subplot(gs[3, 1], sharex=ax_thr)
    ax_dmg = fig.add_subplot(gs[4, 1], sharex=ax_thr)

    # ============== 3D scene ==============
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                    cmap="Blues", alpha=0.55, linewidth=0,
                    antialiased=True, vmin=-2.0, vmax=2.0)

    # Trajectory
    tx = np.array([d["x"] for d in traj])
    tz = np.array([d["z"] for d in traj])
    ty = np.array([d.get("y", 0.0) for d in traj])
    ax.plot(tx, ty, tz, color="tab:orange", lw=2.5, label="flight path")
    mark_idx = np.arange(0, len(traj), max(1, len(traj) // 12))
    ax.scatter(tx[mark_idx], ty[mark_idx], tz[mark_idx],
               color="tab:orange", s=18, depthshade=True)
    # Past trail up to current frame
    past_idx = np.arange(0, time_index + 1)
    ax.plot(tx[past_idx], ty[past_idx], tz[past_idx],
            color="tab:red", lw=3.0, alpha=0.85)

    # Aircraft model
    span, chord = 15.0, 1.5
    wing = _wing_polygon(span, chord) @ rotation.T + offset
    hull = (_hull_polygon(2.6, 0.55) + np.array([0, 0, -0.3])) @ rotation.T + offset

    wing_coll = Poly3DCollection([wing], alpha=0.65,
                                facecolor="steelblue",
                                edgecolor="black", linewidth=1.2)
    ax.add_collection3d(wing_coll)
    chord_line = np.array([[-chord / 2, 0, 0], [chord / 2, 0, 0]]) @ rotation.T + offset
    ax.plot(chord_line[:, 0], chord_line[:, 1], chord_line[:, 2],
            color="red", lw=2.0)
    hull_coll = Poly3DCollection([hull], alpha=0.90,
                                facecolor="dimgray",
                                edgecolor="black", linewidth=1.0)
    ax.add_collection3d(hull_coll)

    ax.plot([x_a, x_a], [y_a, y_a], [z_a, z_a - 8],
            color="gray", lw=1, ls="--", alpha=0.5)
    ax.plot([x_a - 10, x_a + 10], [y_a, y_a], [0, 0],
            color="navy", lw=0.8, alpha=0.4)

    ax.set_xlim(window_x)
    ax.set_ylim(y_a - 30, y_a + 30)
    ax.set_zlim(window_z)
    ax.set_xlabel("Distance, m")
    ax.set_ylabel("Lateral, m")
    ax.set_zlabel("Altitude, m")
    ax.set_title(
        f"t = {info['t']:.1f} s    "
        f"x = {x_a:.1f} m    y = {y_a:.1f} m    z = {z_a:.2f} m    "
        f"Vx = {info['Vx']:.2f} m/s    "
        f"mode = {info.get('mode', 'N/A')}"
    )
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left")

    # ============== Control-input logs (right column) ==============
    t_all = np.array([d["t"] for d in traj])
    thr   = np.array([d["throttle"] for d in traj])
    alpha_arr = np.array([d["alpha"] for d in traj])
    z_all = np.array([d["z"] for d in traj])
    Vx_all = np.array([d["Vx"] for d in traj])
    cmd_alt = np.array([d.get("cmd_alt", np.nan) for d in traj])
    cmd_spd = np.array([d.get("cmd_speed", np.nan) for d in traj])

    # Vertical line at the current frame
    t_now = info["t"]

    # Throttle panel
    ax_thr.plot(t_all, thr, color="tab:red", lw=1.6)
    ax_thr.axvline(t_now, color="k", ls="--", lw=0.8, alpha=0.6)
    ax_thr.scatter([t_now], [info["throttle"]],
                   color="tab:red", s=70, zorder=5, edgecolor="k",
                   linewidth=1.4)
    ax_thr.set_ylim(-0.05, 1.1)
    ax_thr.set_ylabel("Throttle")
    ax_thr.set_title("Autopilot throttle (servo 1, PWM 1000-2000)")
    ax_thr.grid(True, alpha=0.3)
    ax_thr.text(t_now + 0.2, info["throttle"],
                f"{info['throttle']:.2f}", color="tab:red",
                fontsize=9, va="center")

    # Pitch panel
    ax_pit.plot(t_all, np.degrees(alpha_arr), color="tab:blue", lw=1.6)
    ax_pit.axvline(t_now, color="k", ls="--", lw=0.8, alpha=0.6)
    ax_pit.scatter([t_now], [math.degrees(info["alpha"])],
                   color="tab:blue", s=70, zorder=5, edgecolor="k",
                   linewidth=1.4)
    if "bank" in info:
        ax_pit.plot(t_all, [math.degrees(d.get("bank", 0)) for d in traj], label="bank", color="orange")
        ax_pit.legend(fontsize=7)
    ax_pit.set_ylim(-50 if "bank" in info else -20, 50 if "bank" in info else 20)
    ax_pit.set_ylabel("Pitch α, deg")
    ax_pit.set_title("Autopilot pitch (servo 2, elevator)")
    ax_pit.grid(True, alpha=0.3)
    ax_pit.text(t_now + 0.2, math.degrees(info["alpha"]),
                f"{math.degrees(info['alpha']):+.1f}°", color="tab:blue",
                fontsize=9, va="center")

    # Altitude vs setpoint
    ax_alt.plot(t_all, z_all, color="tab:green", lw=1.6, label="altitude z")
    ax_alt.plot(t_all, cmd_alt, color="tab:green", ls="--",
                lw=1.0, label="cmd alt")
    ax_alt.axvline(t_now, color="k", ls="--", lw=0.8, alpha=0.6)
    ax_alt.scatter([t_now], [info["z"]],
                   color="tab:green", s=70, zorder=5, edgecolor="k",
                   linewidth=1.4)
    ax_alt.set_ylabel("Altitude, m")
    ax_alt.set_title("Altitude vs setpoint")
    ax_alt.grid(True, alpha=0.3)
    ax_alt.legend(loc="upper right", fontsize=8)

    # Speed vs setpoint
    ax_spd.plot(t_all, Vx_all, color="tab:purple", lw=1.6, label="Vx")
    ax_spd.plot(t_all, cmd_spd, color="tab:purple", ls="--",
                lw=1.0, label="cmd speed")
    ax_spd.axvline(t_now, color="k", ls="--", lw=0.8, alpha=0.6)
    ax_spd.scatter([t_now], [info["Vx"]],
                   color="tab:purple", s=70, zorder=5, edgecolor="k",
                   linewidth=1.4)
    ax_spd.set_ylabel("Vx, m/s")
    if "airspeed" in info:
        ax_spd.plot(t_all, [d["airspeed"] for d in traj], label="airspeed", color="orange")
        ax_spd.plot(t_all, [d.get("Vy", 0) for d in traj], label="Vy", color="gray")
    ax_spd.set_title("Ground / air speed vs setpoint")
    ax_spd.grid(True, alpha=0.3)
    ax_spd.legend(loc="upper right", fontsize=8)

    # Damage panel (water ingress, spray severity)
    if "water_mass" in info and "T_factor" in info:
        water = np.array([d.get("water_mass", 0.0) for d in traj])
        T_fac = np.array([d.get("T_factor", 1.0) for d in traj])
        spray_sev = np.array([d.get("spray_severity", 0.0) for d in traj])
        slings = np.array([d.get("sling_events", 0) for d in traj])

        ax_dmg.plot(t_all, water, color="tab:cyan", lw=1.6,
                    label="water mass")
        ax_dmg.axhline(0.20, color="orange", ls=":", lw=1.0,
                       label="warning 0.20 kg")
        ax_dmg.axhline(0.50, color="red", ls=":", lw=1.0,
                       label="critical 0.50 kg")
        ax_dmg.set_ylabel("Water, kg")
        ax_dmg.set_xlabel("Time, s")
        ax_dmg.set_title("Hull water ingress + spray load")
        ax_dmg.grid(True, alpha=0.3)
        ax_dmg.legend(loc="upper left", fontsize=7)
        # Twin axis for spray / T-factor
        ax_dmg2 = ax_dmg.twinx()
        ax_dmg2.plot(t_all, T_fac, color="tab:red", lw=1.0,
                     label="thrust factor", alpha=0.7)
        ax_dmg2.plot(t_all, spray_sev, color="tab:olive", lw=1.0,
                     label="cumul. spray", alpha=0.6)
        ax_dmg2.set_ylim(0.0, 1.2)
        ax_dmg2.legend(loc="upper right", fontsize=7)
        ax_dmg2.axvline(t_now, color="k", ls="--", lw=0.8, alpha=0.6)
    else:
        ax_dmg.text(0.5, 0.5, "(no damage model)", ha="center",
                    va="center", transform=ax_dmg.transAxes)
        ax_dmg.set_axis_off()

    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
def render_animated_gif(traj, sea, *, output: str = "results/scene_animated.gif",
                        step: int = 5, fps: int = 12,
                        window_x_range: float = 80.0,
                        window_z_range: tuple[float, float] = (-6.0, 35.0)):
    """Create an animated GIF using Pillow (one PNG per frame).

    Requires `pillow` to be installed.
    """
    from PIL import Image
    import io

    frames = []
    indices = list(range(0, len(traj), step))
    if indices[-1] != len(traj) - 1:
        indices.append(len(traj) - 1)
    print(f"Rendering {len(indices)} frames ...")
    for i, ti in enumerate(indices):
        png_path = f"/tmp/_frame_{i:04d}.png"
        render_scene(traj, sea, time_index=ti,
                     output=png_path,
                     window_x=(traj[ti]["x"] - window_x_range / 2,
                               traj[ti]["x"] + window_x_range / 2),
                     window_z=window_z_range)
        frames.append(Image.open(png_path))
        if i % 10 == 0:
            print(f"  {i}/{len(indices)}")
    # Save as GIF
    duration_ms = int(1000 / fps)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"Saved animated GIF: {output}")
