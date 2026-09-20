"""Run the full drone flying-boat experiment:

  1. print the aircraft design summary
  2. build an irregular ocean (PM spectrum, Hs=1.5 m, Tp=6 s)
  3. simulate the take-off run (water-taxi -> hump -> lift-off -> climb)
  4. simulate the landing  (glide -> touchdown -> water-run)
  5. write a multi-panel PNG figure and a results CSV

Outputs:
    results/takeoff.png     - take-off run telemetry
    results/landing.png     - landing telemetry
    results/wave_field.png  - snapshot of the irregular sea
    results/profile.png     - side-view of the aircraft over waves
    results/summary.txt     - numerical results
    results/timeseries.csv  - merged time series
"""
from __future__ import annotations

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aircraft import Aircraft, RHO, RHO_W, G
from ocean    import Ocean
from dynamics import (simulate_takeoff, simulate_landing,
                      HullContact, HullDrag)

OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------
def wave_snapshot(sea: Ocean):
    """Static snapshot of the wave field + statistics."""
    x = np.linspace(0.0, 200.0, 1001)
    t = np.linspace(0.0, 60.0, 1201)
    eta = sea.eta_field(x, t)
    np.save(f"{OUT}/wave_field.npy", eta)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.pcolormesh(x, t, eta, shading="auto", cmap="seismic",
                       vmin=-1.5, vmax=1.5)
    ax.set_xlabel("Distance, m")
    ax.set_ylabel("Time, s")
    ax.set_title("Irregular sea surface (PM spectrum, Hs=1.5 m, Tp=6 s)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("η, m")
    fig.tight_layout()
    fig.savefig(f"{OUT}/wave_field.png", dpi=130)
    plt.close(fig)


def plot_takeoff(ac: Aircraft, res):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(res.t, res.Vx, label="Vx")
    ax.plot(res.t, res.Vz, label="Vz")
    ax.set_ylabel("Velocity, m/s"); ax.legend()
    ax.set_title("Take-off run -- velocities")
    ax.set_xlim(0, res.t[-1])

    ax = axes[0, 1]
    ax.plot(res.t, res.z, label="CG altitude")
    ax.plot(res.t, res.x, label="Ground track")
    ax.set_ylabel("Position, m"); ax.legend()
    ax.set_title("Altitude & ground track")
    ax.set_xlim(0, res.t[-1])

    ax = axes[1, 0]
    ax.plot(res.t, res.T, label="Thrust")
    ax.plot(res.t, res.L, label="Lift")
    ax.plot(res.t, res.D, label="Aero drag")
    ax.plot(res.t, res.R, label="Total drag")
    ax.plot(res.t, res.N_water, label="N_water")
    ax.axhline(ac.W, color="k", lw=0.5, ls="--", label="Weight")
    ax.set_ylabel("Force, N"); ax.legend(ncol=2, fontsize=8)
    ax.set_title("Forces")
    ax.set_xlim(0, res.t[-1])

    ax = axes[1, 1]
    ax.plot(res.t, res.Vx / ac.V_stall, label="Vx / V_stall")
    ax.axhline(1.0, color="r", lw=0.5, ls="--", label="stall")
    ax.set_ylabel("Speed ratio"); ax.legend()
    ax.set_title("Speed margin")
    ax.set_xlim(0, res.t[-1])

    ax = axes[2, 0]
    # Overlay wave elevation along the track
    eta_track = np.array([
        float(sea.eta(np.array([res.x[i]]), res.t[i])[0])
        for i in range(0, len(res.t), 25)
    ])
    t_track = res.t[::25]
    ax.plot(t_track, eta_track, label="wave η")
    ax.plot(res.t, res.z, label="CG altitude")
    ax.set_xlabel("Time, s"); ax.set_ylabel("z, m"); ax.legend()
    ax.set_title("Wave vs CG altitude")
    ax.set_xlim(0, res.t[-1])

    ax = axes[2, 1]
    ax.plot(res.x, res.z)
    ax.set_xlabel("x, m"); ax.set_ylabel("z, m")
    ax.set_title("Trajectory")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/takeoff.png", dpi=130)
    plt.close(fig)


def plot_landing(ac: Aircraft, res):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    axes[0, 0].plot(res.t, res.Vx); axes[0, 0].plot(res.t, res.Vz)
    axes[0, 0].set_ylabel("Velocity, m/s"); axes[0, 0].set_title("Landing -- velocities")
    axes[0, 0].set_xlim(0, res.t[-1])

    axes[0, 1].plot(res.t, res.z); axes[0, 1].plot(res.t, res.x)
    axes[0, 1].set_ylabel("Position, m"); axes[0, 1].set_title("Altitude & ground track")
    axes[0, 1].set_xlim(0, res.t[-1])

    axes[1, 0].plot(res.t, res.L, label="Lift")
    axes[1, 0].plot(res.t, res.D, label="Aero drag")
    axes[1, 0].plot(res.t, res.N_water, label="N_water")
    axes[1, 0].axhline(ac.W, color="k", lw=0.5, ls="--", label="Weight")
    axes[1, 0].set_ylabel("Force, N"); axes[1, 0].legend()
    axes[1, 0].set_title("Forces")
    axes[1, 0].set_xlim(0, res.t[-1])

    axes[1, 1].plot(res.t, res.Vx / ac.V_stall, label="Vx / V_stall")
    axes[1, 1].axhline(1.0, color="r", lw=0.5, ls="--", label="stall")
    axes[1, 1].set_ylabel("Speed ratio"); axes[1, 1].legend()
    axes[1, 1].set_title("Speed margin")
    axes[1, 1].set_xlim(0, res.t[-1])

    axes[2, 0].plot(res.t, res.z, label="CG altitude")
    axes[2, 0].set_xlabel("Time, s"); axes[2, 0].set_ylabel("z, m")
    axes[2, 0].set_title("Altitude profile")
    axes[2, 0].set_xlim(0, res.t[-1])

    axes[2, 1].plot(res.x, res.z)
    axes[2, 1].set_xlabel("x, m"); axes[2, 1].set_ylabel("z, m")
    axes[2, 1].set_title("Trajectory"); axes[2, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/landing.png", dpi=130)
    plt.close(fig)


def plot_profile(ac: Aircraft, sea: Ocean, res_to, res_ld):
    """Side view: aircraft path over the wave surface."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.linspace(0.0, 420.0, 1001)
    # Build a composite wave elevation: use same seed for repeatability
    eta_takeoff = np.array([
        float(sea.eta(np.array([res_to.x[i]]), res_to.t[i])[0])
        for i in range(0, len(res_to.t), 25)
    ])
    eta_landing = np.array([
        float(sea.eta(np.array([res_ld.x[i]]), res_ld.t[i])[0])
        for i in range(0, len(res_ld.t), 25)
    ])
    ax.plot(res_to.x[::25], eta_takeoff, lw=0.8, color="steelblue",
            alpha=0.6, label="wave η (take-off)")
    ax.plot(res_to.x[::25], res_to.z[::25], color="tab:orange",
            label="take-off path")
    ax.plot(res_ld.x[::25], eta_landing, lw=0.8, color="steelblue",
            alpha=0.4, label="wave η (landing)")
    ax.plot(res_ld.x[::25], res_ld.z[::25], color="tab:green",
            label="landing path")
    # Mark wing span as horizontal scale bar
    ax.plot([0, ac.geom.b], [-3, -3], color="k", lw=3)
    ax.text(ac.geom.b / 2, -3.4, f"wing span = {ac.geom.b:.1f} m",
            ha="center", va="top")
    ax.set_xlabel("Distance, m"); ax.set_ylabel("Altitude, m")
    ax.set_title("Side view :  take-off and landing trajectories over irregular sea")
    ax.set_xlim(0, max(res_ld.x[-1], 410))
    ax.set_ylim(-4, max(max(res_to.z), max(res_ld.z)) + 5)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/profile.png", dpi=130)
    plt.close(fig)


def write_summary(ac, sea, res_to, res_ld):
    # --- take-off metrics ---
    liftoff_idx = int(np.argmax(res_to.phase))
    lift_to = res_to.t[liftoff_idx]
    x_lo    = res_to.x[liftoff_idx]
    V_lo    = res_to.Vx[liftoff_idx]
    R_peak  = res_to.R.max()
    V_top   = res_to.Vx.max()

    # --- landing metrics ---
    contact_idx = np.where(res_ld.N_water > 0)[0]
    if len(contact_idx) > 0:
        td = int(contact_idx[0])
        td_t   = res_ld.t[td]
        td_x   = res_ld.x[td]
        td_Vx  = res_ld.Vx[td]
        td_Vz  = res_ld.Vz[td]
        N_pk   = res_ld.N_water.max()
        V_end  = res_ld.Vx[-1]
        slow = np.where(res_ld.Vx < 0.2)[0]
        stop_idx = int(slow[0]) if len(slow) > 0 else -1
        stop_t = res_ld.t[stop_idx] if stop_idx >= 0 else float("nan")
        stop_x = res_ld.x[stop_idx] if stop_idx >= 0 else float("nan")
    else:
        td_t = td_x = td_Vx = td_Vz = N_pk = V_end = stop_t = stop_x = float("nan")

    # --- sea statistics ---
    stats = sea.statistics()

    txt = []
    txt.append("Drone flying-boat experiment -- results")
    txt.append("=" * 60)
    txt.append(ac.summary())
    txt.append("")
    txt.append("Ocean statistics (Monte-Carlo over 600 s)")
    txt.append("-" * 60)
    for k, v in stats.items():
        txt.append(f"  {k:15s} : {v:.3f}")
    txt.append("")
    txt.append("Take-off metrics")
    txt.append("-" * 60)
    txt.append(f"  Lift-off time        : {lift_to:.2f} s")
    txt.append(f"  Lift-off ground track: {x_lo:.1f} m")
    txt.append(f"  Lift-off speed (Vx)  : {V_lo:.2f} m/s ({V_lo/ac.V_stall:.2f} V_stall)")
    txt.append(f"  Top speed reached    : {V_top:.2f} m/s")
    txt.append(f"  Peak total drag      : {R_peak:.0f} N ({R_peak/ac.W*100:.0f} % of W)")
    txt.append("")
    txt.append("Landing metrics")
    txt.append("-" * 60)
    txt.append(f"  Touchdown time       : {td_t:.2f} s")
    txt.append(f"  Touchdown ground trk : {td_x:.1f} m")
    txt.append(f"  Touchdown Vx / |Vz|  : {td_Vx:.2f} / {abs(td_Vz):.2f} m/s")
    txt.append(f"  Peak water normal f. : {N_pk:.0f} N ({N_pk/ac.W:.2f} x W)")
    txt.append(f"  Final speed          : {V_end:.2f} m/s")
    txt.append(f"  Stop time / distance : {stop_t:.2f} s, {stop_x:.1f} m")
    txt.append("")
    out = "\n".join(txt)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(out)
    return out


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Aircraft
    ac = Aircraft()
    print(ac.summary())

    # 2. Ocean
    sea = Ocean(Hs=1.5, Tp=6.0, seed=42)
    print("\nWave statistics:")
    for k, v in sea.statistics().items():
        print(f"  {k:15s} : {v:.3f}")

    # 3. Take-off run
    res_to = simulate_takeoff(
        ac, sea,
        duration=12.0, dt=0.001,
        throttle=1.0, alpha=math.radians(4.0),
    )
    plot_takeoff(ac, res_to)

    # 4. Landing
    res_ld = simulate_landing(
        ac, sea,
        approach_alt=30.0, approach_speed=13.0,
        glide_slope=math.radians(8.0),
        alpha_body=math.radians(-5.0),
        throttle=0.05,
        duration=40.0, dt=0.001,
    )
    plot_landing(ac, res_ld)

    # 5. Wave field
    wave_snapshot(sea)

    # 6. Side-view profile
    plot_profile(ac, sea, res_to, res_ld)

    # 7. CSV time series
    # Use only the take-off series for the CSV (they use the same dt but
    # different durations)
    np.savetxt(
        f"{OUT}/timeseries_takeoff.csv",
        np.column_stack([res_to.t, res_to.x, res_to.z,
                         res_to.Vx, res_to.Vz, res_to.T, res_to.L,
                         res_to.D, res_to.R, res_to.N_water]),
        delimiter=",",
        header="t,x,z,Vx,Vz,T,L,D,R,N_water",
    )
    np.savetxt(
        f"{OUT}/timeseries_landing.csv",
        np.column_stack([res_ld.t, res_ld.x, res_ld.z,
                         res_ld.Vx, res_ld.Vz, res_ld.T, res_ld.L,
                         res_ld.D, res_ld.R, res_ld.N_water]),
        delimiter=",",
        header="t,x,z,Vx,Vz,T,L,D,R,N_water",
    )

    # 8. Summary
    summary = write_summary(ac, sea, res_to, res_ld)
    print()
    print(summary)
    print(f"\nAll outputs written to {OUT}/")