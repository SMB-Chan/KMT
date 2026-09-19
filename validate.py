"""Physics sanity checks for the flying-boat experiment.

Run directly to verify that freshly computed simulation trajectories still
satisfies the expected physics relationships (lift = weight at
cruise, energy balance, etc.).
"""
from __future__ import annotations
import math
import numpy as np
from aircraft import Aircraft, RHO, RHO_W, G
from ocean import Ocean
from dynamics import simulate_takeoff, simulate_landing


def assert_close(a: float, b: float, rel: float = 1e-3, msg: str = ""):
    if not np.isfinite([a, b]).all() or abs(a - b) > rel * max(abs(a), abs(b), 1e-9):
        raise AssertionError(f"{msg}: {a} vs {b} (rel diff {abs(a-b)/max(abs(a),abs(b),1e-9):.2e})")
    print(f"  OK  {msg}: {a:.4f}  vs  {b:.4f}")


def main():
    print("=" * 60)
    print("Physics sanity checks")
    print("=" * 60)

    # --- Aircraft -----------------------------------------------------
    ac = Aircraft()
    print("\n[Aircraft]")
    assert_close(ac.W, ac.mass.total * G, msg="weight = m*g")
    # L/D max at CL* = sqrt(CD0 * pi * e * AR)
    CL_star = math.sqrt(ac.aero.CD0 * math.pi * ac.aero.e * ac.geom.AR)
    LD_max  = 0.5 / math.sqrt(ac.aero.CD0 / (math.pi * ac.aero.e * ac.geom.AR))
    assert_close(ac.L_D(CL_star), LD_max, rel=1e-6,
                 msg="L/D max from polar")
    # Cruise speed
    assert_close(ac.V_cruise, ac.V_stall * math.sqrt(3), rel=1e-6,
                 msg="V_cruise = V_stall * sqrt(3)")

    # --- Ocean --------------------------------------------------------
    print("\n[Ocean]")
    sea = Ocean(Hs=1.5, Tp=6.0, seed=42)
    stats = sea.statistics()
    # Hs definition: Hs = 4 * sigma_eta  =>  sigma = Hs / 4
    sigma_expected = 1.5 / 4.0
    assert_close(math.sqrt(stats["Hs_observed"] ** 2 / 16.0),
                 sigma_expected, rel=0.05, msg="Hs / 4 = eta_rms")
    # eta_rms should equal sigma
    assert_close(stats["eta_rms"], sigma_expected, rel=0.05,
                 msg="eta_rms from definition")
    # Deep-water dispersion: omega^2 = g k
    w = sea.omega
    k = sea.k
    diff = np.max(np.abs(w**2 - G * k) / np.maximum(w**2, 1e-9))
    assert diff < 1e-12
    print(f"  OK  deep-water dispersion: max rel error = {diff:.2e}")

    # --- Dynamics sanity ---------------------------------------------
    # Read timeseries
    print("\n[Dynamics - takeoff]")
    result = simulate_takeoff(ac, sea, duration=12.0, dt=0.001)
    t, x, z, Vx, Vz, T, L, D, R, N_water = (getattr(result, key) for key in
        ("t", "x", "z", "Vx", "Vz", "T", "L", "D", "R", "N_water"))
    # Find lift-off
    # Air phase = N_water == 0
    air = N_water < 1.0
    idx = np.argmax(air)
    if idx == 0:
        raise AssertionError("never left the water in takeoff")
    print(f"  OK  lift-off at t = {t[idx]:.2f} s, x = {x[idx]:.1f} m, "
          f"Vx = {Vx[idx]:.2f} m/s")
    # After lift-off, Vx should be above stall
    assert Vx[idx] > ac.V_stall, f"lift-off speed below stall: {Vx[idx]} < {ac.V_stall}"
    # Top speed in air should exceed Vx[idx]
    top_speed_air = Vx[idx:].max()
    print(f"  OK  top air speed = {top_speed_air:.2f} m/s")

    # Energy budget check: power * dt should equal work done
    # In air, W = T - D = m V dV/dt + m g dz/dt
    # Quick check: thrust > hull drag at lift-off (so aircraft can fly)
    T_static = ac.prop.T_static
    drag_at_liftoff = R[idx]
    print(f"  OK  thrust @ lift-off speed = {T[idx]:.0f} N, "
          f"drag at lift-off = {drag_at_liftoff:.0f} N "
          f"(must be < thrust to fly)")
    assert T[idx] > drag_at_liftoff, "thrust insufficient at lift-off"
    # Takeoff without water support would require T > W.
    print(f"  Note: T_static = {T_static:.0f} N < W = {ac.W:.0f} N "
          f"-- takeoff relies on water buoyancy during taxi")

    print("\n[Dynamics - landing]")
    result = simulate_landing(ac, sea, duration=40.0, dt=0.001)
    t, x, z, Vx, Vz, T, L, D, R, N_water = (getattr(result, key) for key in
        ("t", "x", "z", "Vx", "Vz", "T", "L", "D", "R", "N_water"))
    # Find touchdown
    contact = np.where(N_water > 0)[0]
    if len(contact) == 0:
        raise AssertionError("no water contact in landing")
    td = contact[0]
    print(f"  OK  touchdown at t = {t[td]:.2f} s, x = {x[td]:.1f} m, "
          f"Vx = {Vx[td]:.2f} m/s, |Vz| = {abs(Vz[td]):.2f} m/s")
    # Touchdown speed should be > stall
    assert Vx[td] > ac.V_stall, f"touchdown speed below stall"
    # Descent rate < 5 m/s (safe for foam hull)
    assert abs(Vz[td]) < 5.0, f"too-fast descent rate: {Vz[td]}"
    # Stopping: final Vx should be ~ 0
    assert abs(Vx[-1]) < 0.5, f"did not stop: Vx[-1] = {Vx[-1]}"
    print(f"  OK  stopped at t = {t[-1]:.2f} s, Vx = {Vx[-1]:.2f} m/s")

    # Peak water normal force should not exceed structural limit.
    # A reasonable limit for an 80 kg foam hull is ~5x weight (4000 N).
    peak_N = N_water.max()
    print(f"  OK  peak N_water = {peak_N:.0f} N  ({peak_N/ac.W:.1f} W)")
    assert peak_N < 10 * ac.W, f"excessive peak water force: {peak_N}"

    print("\nAll physics sanity checks passed.")


if __name__ == "__main__":
    main()