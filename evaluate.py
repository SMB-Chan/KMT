"""End-to-end evaluation of the scripted autopilot using the MAVLink
interface and the real-ocean model, with 3D visualisation.

Runs:
    1. MAV_CMD_NAV_TAKEOFF  via FlyingBoatVehicle
    2. MAV_CMD_NAV_LAND     via FlyingBoatVehicle
and produces:
    * 3D static scenes (initial, mid, final) with control-input logs
    * side-view profile plot
    * MAVLink telemetry CSV
"""
from __future__ import annotations

import math
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aircraft  import Aircraft
from ocean_real import load_buoy_default
from mavlink_if import (FlyingBoatVehicle,
                        MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_LAND,
                        MAV_CMD_NAV_WAYPOINT, MAV_CMD_DO_CHANGE_SPEED,
                        MAV_CMD_DO_SET_SERVO)
from visual3d  import render_scene

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def _build_trajectory_from_vehicle(veh: FlyingBoatVehicle, sea):
    """Build a flat dict-list of (state, command) history from a vehicle.

    The vehicle records the *state* at the time of telemetry emission, so
    we snapshot veh.{x,z,Vx,Vz,alpha,throttle,mode} at each step.
    """
    traj = []
    for entry in veh._msg_log:
        # new format: (t, hil, gpi, snap); old format fallback: (t, hil, gpi)
        if len(entry) == 4:
            t, hil, gpi, snap = entry
        else:
            t, hil, gpi = entry
            snap = dict(x=getattr(veh, "x", 0.0), z=getattr(veh, "z", 0.0),
                        Vx=getattr(veh, "Vx", 0.0), Vz=getattr(veh, "Vz", 0.0),
                        alpha=getattr(veh, "alpha", 0.0),
                        throttle=getattr(veh, "throttle", 0.0),
                        mode=getattr(veh, "_mode", "N/A"))
        d = dict(
            t=t,
            x=float(snap["x"]), z=float(snap["z"]),
            Vx=float(snap["Vx"]), Vz=float(snap["Vz"]),
            alpha=float(snap["alpha"]),
            throttle=float(snap["throttle"]),
            eta=float(np.asarray(sea.eta(np.array([snap["x"]]), t)).ravel()[0]),
            mode=str(snap.get("mode", "N/A")),
            # Damage info (optional)
            water_mass=float(snap.get("water_mass", 0.0)),
            T_factor=float(snap.get("T_factor", 1.0)),
            spray_severity=float(snap.get("spray_severity", 0.0)),
            sling_events=int(snap.get("sling_events", 0)),
            damage_status=str(snap.get("damage_status", "OK")),
        )
        # Command setpoints (constant for the active command)
        if veh._active_cmd == MAV_CMD_NAV_TAKEOFF:
            p = veh._cmd_params or {}
            d["cmd_alt"]   = float(p.get("alt", np.nan))
            d["cmd_speed"] = float(p.get("speed", np.nan))
        elif veh._active_cmd == MAV_CMD_NAV_LAND:
            p = veh._cmd_params or {}
            d["cmd_alt"]   = float(p.get("alt", np.nan))
            d["cmd_speed"] = np.nan
        else:
            d["cmd_alt"]   = np.nan
            d["cmd_speed"] = np.nan
        traj.append(d)
    return traj


def _save_telemetry_csv(veh: FlyingBoatVehicle, filename: str):
    """Save MAVLink-style telemetry to CSV."""
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "msg", "x", "y", "z", "Vx", "Vz", "roll",
                    "pitch", "yaw", "alt_mm", "mode"])
        for entry in veh._msg_log:
            if len(entry) == 4:
                t, hil, gpi, snap = entry
            else:
                t, hil, gpi = entry
                snap = dict(x=veh.x, z=veh.z, Vx=veh.Vx, Vz=veh.Vz,
                            alpha=veh.alpha, throttle=veh.throttle,
                            mode=veh._mode)
            w.writerow([t, "HIL_STATE", snap["x"], 0.0, snap["z"],
                        snap["Vx"], snap["Vz"], 0.0,
                        math.degrees(snap["alpha"]), 0.0,
                        int(snap["z"] * 1000), snap["mode"]])
        # Comment lines stay parseable: pandas read_csv(comment="#")
        # skips them while the data rows keep the 12-column header shape.
        w.writerow([])
        w.writerow(["# summary end_t", veh.t, "final_z", veh.z, "final_x", veh.x])


def run_takeoff():
    print("=" * 60)
    print("MAVLink-driven take-off, real NDBC ocean")
    print("=" * 60)
    ac = Aircraft()
    sea = load_buoy_default()
    veh = FlyingBoatVehicle(ac, sea)
    veh.reset()
    veh.arm()
    veh.send_command(MAV_CMD_NAV_TAKEOFF, {"alt": 10.0, "speed": 13.0})
    print(f"Issued: MAV_CMD_NAV_TAKEOFF(alt=10, speed=13)  -- NDBC: {sea.label}")

    duration = 12.0
    n = int(duration / veh.dt)
    for i in range(n):
        veh.step()

    print(f"  Final state: t={veh.t:.2f} s  x={veh.x:.1f} m  "
          f"z={veh.z:.2f} m  Vx={veh.Vx:.2f} m/s")
    print(f"  Telemetry messages logged: {len(veh._msg_log)}")

    # Build trajectory and render
    traj = _build_trajectory_from_vehicle(veh, sea)
    _save_telemetry_csv(veh, f"{OUT}/mavlink_takeoff_telemetry.csv")

    # Three 3D scenes at different time slices
    for label, idx in [("early", n // 6), ("mid", n // 2), ("final", -1)]:
        render_scene(traj, sea, time_index=idx,
                     output=f"{OUT}/scene_takeoff_{label}.png")
    print(f"  3D scenes saved: scene_takeoff_early/mid/final.png")

    # Side profile (2-D) with damage panel
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot([d["t"] for d in traj], [d["z"] for d in traj], label="z")
    ax.plot([d["t"] for d in traj], [d["cmd_alt"] for d in traj],
            ls="--", label="cmd alt")
    ax.axhline(10.0, color="r", lw=0.5, ls=":", label="goal 10 m")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Altitude, m")
    ax.set_title("Take-off -- altitude"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot([d["t"] for d in traj], [d["Vx"] for d in traj], label="Vx")
    ax.plot([d["t"] for d in traj], [d["cmd_speed"] for d in traj],
            ls="--", label="cmd speed")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Vx, m/s")
    ax.set_title("Take-off -- forward speed"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot([d["t"] for d in traj], [d["throttle"] for d in traj],
            color="tab:red", label="throttle")
    ax.plot([d["t"] for d in traj], np.degrees([d["alpha"] for d in traj]),
            color="tab:blue", label="pitch α, deg")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Cmd")
    ax.set_title("Autopilot outputs"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    tx = [d["x"] for d in traj]; tz = [d["z"] for d in traj]
    eta_x = [float(np.asarray(sea.eta(np.array([x]), t)).ravel()[0])
             for x, t in zip(tx, [d["t"] for d in traj])]
    ax.plot(tx, eta_x, color="steelblue", alpha=0.5, label="wave η")
    ax.plot(tx, tz, color="tab:orange", lw=2, label="flight path")
    ax.set_xlabel("Distance, m"); ax.set_ylabel("Altitude, m")
    ax.set_title("Take-off profile (real NDBC sea)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Damage panel
    ax = axes[2, 0]
    ax.plot([d["t"] for d in traj], [d.get("water_mass", 0.0) for d in traj],
            color="tab:cyan", label="water mass")
    ax.axhline(0.20, color="orange", ls=":", lw=1.0, label="warning")
    ax.axhline(0.50, color="red",    ls=":", lw=1.0, label="critical")
    ax.set_xlabel("Time, s"); ax.set_ylabel("kg")
    ax.set_title("Hull water ingress"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot([d["t"] for d in traj], [d.get("T_factor", 1.0) for d in traj],
            color="tab:red", label="thrust factor")
    ax.plot([d["t"] for d in traj], [d.get("spray_severity", 0.0) for d in traj],
            color="tab:olive", label="cumul. spray")
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel("Time, s"); ax.set_ylabel("factor")
    ax.set_title("Propeller spray load"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/mavlink_takeoff_profile.png", dpi=130)
    plt.close(fig)
    print(f"  2-D profile saved: mavlink_takeoff_profile.png")
    return traj, veh


def run_landing():
    print("\n" + "=" * 60)
    print("MAVLink-driven landing, real NDBC ocean")
    print("=" * 60)
    ac = Aircraft()
    sea = load_buoy_default()
    veh = FlyingBoatVehicle(ac, sea)
    # Initial condition: altitude 25 m, glide slope 8 deg
    gs = math.radians(8.0)
    veh.x = 0.0
    veh.z = 25.0
    veh.Vx = 13.0 * math.cos(gs)
    veh.Vz = -13.0 * math.sin(gs)
    veh.t = 0.0
    veh._msg_log = []
    veh.arm()
    veh.send_command(MAV_CMD_NAV_LAND, {"alt": 0.0, "glide": 8.0})
    print(f"Issued: MAV_CMD_NAV_LAND(alt=0, glide=8°)")

    duration = 30.0
    n = int(duration / veh.dt)
    for i in range(n):
        veh.step()
        if veh.damage.failed:
            break

    print(f"  Final state: t={veh.t:.2f} s  x={veh.x:.1f} m  "
          f"z={veh.z:.2f} m  Vx={veh.Vx:.2f} m/s")
    print(f"  Telemetry messages logged: {len(veh._msg_log)}")

    traj = _build_trajectory_from_vehicle(veh, sea)
    _save_telemetry_csv(veh, f"{OUT}/mavlink_landing_telemetry.csv")

    for label, idx in [("early", n // 6), ("mid", n // 2), ("final", -1)]:
        idx = min(idx, len(traj) - 1)
        render_scene(traj, sea, time_index=idx,
                     output=f"{OUT}/scene_landing_{label}.png")
    print(f"  3D scenes saved: scene_landing_early/mid/final.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    ax = axes[0, 0]
    ax.plot([d["t"] for d in traj], [d["z"] for d in traj], label="z")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Altitude, m")
    ax.set_title("Landing -- altitude"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot([d["t"] for d in traj], [d["Vx"] for d in traj], label="Vx")
    ax.plot([d["t"] for d in traj], [d["Vz"] for d in traj], label="Vz")
    ax.set_xlabel("Time, s"); ax.set_ylabel("V, m/s")
    ax.set_title("Landing -- velocities"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot([d["t"] for d in traj], [d["throttle"] for d in traj],
            color="tab:red", label="throttle")
    ax.plot([d["t"] for d in traj], np.degrees([d["alpha"] for d in traj]),
            color="tab:blue", label="pitch α, deg")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Cmd")
    ax.set_title("Autopilot outputs"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    tx = [d["x"] for d in traj]; tz = [d["z"] for d in traj]
    eta_x = [float(np.asarray(sea.eta(np.array([x]), t)).ravel()[0])
             for x, t in zip(tx, [d["t"] for d in traj])]
    ax.plot(tx, eta_x, color="steelblue", alpha=0.5, label="wave η")
    ax.plot(tx, tz, color="tab:green", lw=2, label="flight path")
    ax.set_xlabel("Distance, m"); ax.set_ylabel("Altitude, m")
    ax.set_title("Landing profile (real NDBC sea)")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/mavlink_landing_profile.png", dpi=130)
    plt.close(fig)
    print(f"  2-D profile saved: mavlink_landing_profile.png")
    return traj, veh


if __name__ == "__main__":
    run_takeoff()
    run_landing()
    print("\nDone.  Outputs in", OUT)