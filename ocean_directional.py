"""2-D irregular ocean with directional spreading.

The classical way to add direction to a 1-D wave spectrum is to use a
directional spreading function D(theta).  The Mitsuyasu / Longuet-
Higgins form used here is

    D(theta; s) = C(s) cos^{2 s}((theta - theta_mean) / 2)

where `s` controls the width (large s = narrow beam, s ~ 4 for wind
sea, s ~ 25 for mature swell).  C(s) normalises the function so that
the integrated energy in the frequency spectrum is unchanged.

The 2-D free-surface elevation is then

    eta(x, y, t) = sum_{i, j}
        a_{ij} * cos( k_i ( x cos(theta_j) + y sin(theta_j) )
                    - omega_i t
                    + phi_{ij} )

with the dispersion relation omega^2 = g k tanh(k h)  (deep-water
approximation omega^2 = g k).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from ocean import pm_spectrum, G


# ---------------------------------------------------------------------
def cos2s_spreading(theta: np.ndarray, theta_mean: float,
                    s: float) -> np.ndarray:
    """Longuet-Higgins cos^{2s} directional spreading function.

    Returns a vector that integrates to 1 over the supplied theta grid.
    """
    arg = (theta - theta_mean) / 2.0
    D = np.cos(arg) ** (2 * s)
    # Avoid negative values from round-off
    D = np.clip(D, 0.0, None)
    dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0
    norm = D.sum() * dtheta
    if norm <= 0:
        # Fall back to uniform
        return np.ones_like(theta) / (2.0 * math.pi)
    return D / norm


# ---------------------------------------------------------------------
@dataclass
class DirectionalOcean:
    """2-D sea surface with directional spreading.

    Parameters
    ----------
    Hs        : significant wave height (m)
    Tp        : peak period (s)
    theta_mean: mean wave direction (rad, measured CCW from +x axis)
    s         : spreading parameter (cos^{2s}); larger = narrower beam
    n_freq    : number of frequency components
    n_dir     : number of directional components
    depth     : water depth (m); default 50 m (deep-water approx holds)
    seed      : RNG seed
    """
    Hs: float = 1.5
    Tp: float = 6.0
    theta_mean: float = 0.0
    s: int = 10
    n_freq: int = 32
    n_dir: int = 16
    omega_lo: float = 0.2
    omega_hi: float = 4.0
    depth: float = 50.0
    seed: int = 42

    # internal
    omega: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)
    theta: np.ndarray = field(init=False, repr=False)
    cos_t: np.ndarray = field(init=False, repr=False)
    sin_t: np.ndarray = field(init=False, repr=False)
    amps: np.ndarray = field(init=False, repr=False)
    phases: np.ndarray = field(init=False, repr=False)
    domega: float = field(init=False, repr=False)
    dtheta: float = field(init=False, repr=False)
    S_1d: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Frequency grid
        self.omega = np.linspace(self.omega_lo, self.omega_hi, self.n_freq)
        self.domega = self.omega[1] - self.omega[0]
        # Direction grid spanning [-pi, pi]
        self.theta = np.linspace(-math.pi, math.pi, self.n_dir,
                                 endpoint=False) + math.pi / self.n_dir
        self.dtheta = self.theta[1] - self.theta[0]
        # Deep-water dispersion
        self.k = self.omega ** 2 / G
        # 1-D spectrum (m^2 / (rad/s))
        self.S_1d = pm_spectrum(self.omega, self.Hs, self.Tp)
        # Directional spreading
        D = cos2s_spreading(self.theta, self.theta_mean, self.s)
        # Per-(freq, dir) amplitudes:  a^2 = 2 S(omega) D(theta) domega dtheta
        amp2 = 2.0 * self.S_1d[:, None] * D[None, :] * self.domega * self.dtheta
        self.amps = np.sqrt(amp2)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
        # Random phases per component
        self.phases = rng.uniform(0.0, 2.0 * math.pi,
                                  size=(self.n_freq, self.n_dir))

    # ----- 1-D longitudinal slice (for dynamics) -----
    def eta_long(self, x, t):
        """Evaluate eta at (x, y=0) along the mean wave direction."""
        x = np.atleast_1d(x)
        t = np.atleast_1d(t)
        # k[i, j] * x * cos(theta[j]): shape (n_freq, n_dir, Nt, Nx)
        arg = (self.k[:, None, None, None] * x[None, None, None, :]
               * self.cos_t[None, :, None, None]
               - self.omega[:, None, None, None] * t[None, None, :, None]
               + self.phases[:, :, None, None])
        # eta[t, x] = sum_{i, j} a[i, j] cos(arg[i, j, t, x])
        eta = (self.ams_broadcast(x, t) * np.cos(arg)).sum(axis=(0, 1))
        return eta  # shape (Nt, Nx) for 2-D inputs, else scalar

    # ----- full 2-D evaluation -----
    def eta(self, x, y, t):
        """Evaluate the 2-D wave field.

        x, y : 1-D arrays of positions
        t    : scalar time  OR  1-D array of times

        Returns
        -------
        eta : ndarray
            * If t is scalar:  shape (Ny, Nx)
            * If t is 1-D:    shape (Nt, Ny, Nx)
        """
        x = np.atleast_1d(x); y = np.atleast_1d(y)
        if np.isscalar(t):
            t_arr = np.array([float(t)])
            scalar_t = True
        else:
            t_arr = np.atleast_1d(t)
            scalar_t = False
        X, Y, T = np.meshgrid(x, y, t_arr, indexing="xy")     # (Ny, Nx, Nt)
        # Broadcast component indices
        # k[i, j, ny, nx, nt]
        kxy = (self.k[:, None, None, None, None]
               * (X[None, :, :, :] * self.cos_t[None, :, None, None, None]
                  + Y[None, :, :, :] * self.sin_t[None, :, None, None, None]))
        wt = self.omega[:, None, None, None, None] * T[None, :, :, :]
        arg = kxy - wt + self.phases[:, :, None, None, None]
        amp_b = self.ams_broadcast_3d(X.shape)
        eta = (amp_b * np.cos(arg)).sum(axis=(0, 1))
        # eta has shape (Ny, Nx, Nt); transpose to (Nt, Ny, Nx)
        eta = np.transpose(eta, (2, 0, 1))
        if scalar_t:
            eta = eta[0]
        return eta

    def eta_1d(self, x, t):
        """Longitudinal slice (y=0) with Ocean.eta-compatible convention.

        x : array-like positions; t : scalar time.
        Returns ndarray of shape (Nx,).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        eta = self.eta_long(x, np.array([float(t)]))
        return np.asarray(eta).reshape(-1)

    # ----- helper for amplitude broadcasting -----
    def ams_broadcast(self, x_arr, t_arr):
        """Shape (n_freq, n_dir, Nt, Nx) of amplitudes."""
        Nt = len(t_arr); Nx = len(x_arr)
        return (self.amps[:, :, None, None]
                * np.ones((self.n_freq, self.n_dir, Nt, Nx)))

    def ams_broadcast_3d(self, shape_3d):
        """shape (n_freq, n_dir, Ny, Nx, Nt) for full 2-D evaluation."""
        return (self.amps[:, :, None, None, None]
                * np.ones((self.n_freq, self.n_dir) + tuple(shape_3d)))

    # ----- statistics -----
    def statistics(self, x0: float = 0.0, y0: float = 0.0,
                   t: np.ndarray | None = None,
                   n_samples: int = 20000) -> dict:
        if t is None:
            t = np.linspace(0.0, 600.0, n_samples)
        eta = self.eta_long(np.array([x0]), t)
        if eta.ndim == 2:
            eta = eta[:, 0]
        eta_dot = np.gradient(eta, t)
        m0 = float(np.mean(eta ** 2))
        m2 = float(np.mean(eta_dot ** 2))
        Tz = 2 * math.pi * math.sqrt(m0 / max(m2, 1e-12))
        return {
            "Hs_target":       self.Hs,
            "Hs_observed":     4.0 * math.sqrt(m0),
            "Tp_target":       self.Tp,
            "Tz_observed":     Tz,
            "theta_mean_deg":  math.degrees(self.theta_mean),
            "spread_s":        self.s,
            "eta_rms":         math.sqrt(m0),
            "crest_max":       float(eta.max()),
            "trough_min":      float(eta.min()),
        }


# ---------------------------------------------------------------------
#  Real-data wrapper: uses NDBC MWD when available
# ---------------------------------------------------------------------
def from_ndbc(realtime_path="data/ndbc_46012_realtime.txt",
              spectral_path="data/ndbc_46026_spectral.txt",
              s: int = 8,
              n_dir: int = 16):
    """Build a DirectionalOcean from NDBC files.

    Uses the latest MWD value (mean wave direction) when available.
    """
    import os
    import math
    theta_deg = 0.0   # default = +x
    mwd_raw = None
    Hs = 1.5; Tp = 6.0
    # Spectral file columns (see the '#' header):
    #   YY MM DD hh mm WVHT SwH SwP WWH WWP SwD WWD STEEPNESS APD MWD
    if os.path.exists(spectral_path):
        with open(spectral_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    Hs_sw = float(parts[6]); T_sw = float(parts[7])
                    Hs_ww = float(parts[8]); T_ww = float(parts[9])
                except ValueError:
                    continue
                if Hs_sw <= 0 and Hs_ww <= 0:
                    continue
                # Use the dominant component for Hs/Tp
                if Hs_sw >= Hs_ww:
                    Hs, Tp = Hs_sw, T_sw
                else:
                    Hs, Tp = Hs_ww, T_ww
                # MWD is the last header column; compass-point fields
                # (SwD/WWD) precede it and are not numeric.
                try:
                    mwd_deg_str = parts[14] if len(parts) >= 15 else parts[-1]
                    mwd_deg = (float(mwd_deg_str)
                               if mwd_deg_str != "MM" else None)
                except ValueError:
                    mwd_deg = None
                if mwd_deg is not None and mwd_deg > 0:
                    # MWD is "from" direction (oceanographic convention);
                    # travel bearing is the opposite compass point. Our axes
                    # are x=north, y=east, so a compass bearing numerically
                    # equals the math angle CCW from +x.
                    theta_deg = (mwd_deg + 180.0) % 360.0
                    mwd_raw = mwd_deg
                break
    # Realtime file columns (see the '#' header):
    #   YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ...
    elif os.path.exists(realtime_path):
        with open(realtime_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 11:
                    continue
                try:
                    Hs = float(parts[8]); Tp = float(parts[9])
                except ValueError:
                    continue
                if Hs <= 0 or Tp <= 0:
                    continue
                if len(parts) >= 12:
                    try:
                        mwd_deg_str = parts[11]
                        mwd_deg = (float(mwd_deg_str)
                                   if mwd_deg_str != "MM" else None)
                    except ValueError:
                        mwd_deg = None
                    if mwd_deg is not None and mwd_deg > 0:
                        theta_deg = (mwd_deg + 180.0) % 360.0
                        mwd_raw = mwd_deg
                break
    theta_mean = math.radians(theta_deg)
    mwd_str = f"{mwd_raw:.0f} deg from" if mwd_raw is not None else "unknown"
    label = (f"NDBC Hs={Hs:.2f} m  Tp={Tp:.2f} s  "
             f"MWD={mwd_str}  s={s}")
    sea = DirectionalOcean(Hs=Hs, Tp=Tp, theta_mean=theta_mean, s=s,
                           n_dir=n_dir, seed=42)
    sea.label = label
    return sea


# ---------------------------------------------------------------------
#  1-D longitudinal accessor shared by dynamics, env, and vehicle
# ---------------------------------------------------------------------
def sea_eta_1d(sea, x, t):
    """Elevation with Ocean.eta-compatible convention.

    x : array-like positions; t : scalar time.
    Returns ndarray of shape (Nx,) for Ocean, RealOcean,
    and DirectionalOcean (longitudinal slice at y=0).
    """
    if isinstance(sea, DirectionalOcean):
        return sea.eta_1d(x, t)
    return np.atleast_1d(sea.eta(x, t))


PREVIEW_DX_M = (5.0, 15.0, 30.0)
V_PREVIEW_FLOOR = 1.0


def wave_preview(sea, x, t, vx, dxs=PREVIEW_DX_M):
    """Encounter-time eta at distances ahead of the hull.

    For each look-ahead dx, evaluate eta at the point the hull would
    reach if it kept speed max(|vx|, V_PREVIEW_FLOOR) along the track.
    """
    dxs = np.atleast_1d(np.asarray(dxs, dtype=float))
    vx = float(vx)
    sign = 1.0 if vx >= 0.0 else -1.0
    vx_eff = max(abs(vx), V_PREVIEW_FLOOR)
    x0 = float(x)
    t0 = float(t)
    out = np.empty(dxs.shape[0], dtype=np.float64)
    for i, dx in enumerate(dxs):
        out[i] = float(sea_eta_1d(sea, np.array([x0 + sign * float(dx)]),
                                  t0 + float(dx) / vx_eff)[0])
    return out


# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Directional ocean (2-D + spreading)")
    print("=" * 60)
    # Quick example
    sea = DirectionalOcean(Hs=1.5, Tp=6.0, theta_mean=0.0, s=10,
                           n_freq=32, n_dir=16, seed=42)
    print(f"  Hs = {sea.Hs} m, Tp = {sea.Tp} s, theta_mean = {sea.theta_mean} rad")
    print(f"  Spreading exponent 2s = {2 * sea.s}  (cos^{2*sea.s})")
    stats = sea.statistics()
    for k, v in stats.items():
        print(f"  {k:18s}: {v:.3f}")

    print("\nReal-data version (NDBC 46026):")
    sea_real = from_ndbc()
    print(f"  {sea_real.label}")
    stats = sea_real.statistics()
    for k, v in stats.items():
        print(f"  {k:18s}: {v:.3f}")