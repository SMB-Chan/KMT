"""Wave patch computation and binary encoding.

The cockpit needs a heightmap around the vehicle so the sea surface
renders without an uninitialised neighbour. Per design section 8:

    * default 128 m square, 65x65 grid (~17 KiB per frame at 10 Hz)
    * float32 little-endian, packed as binary
    * origin (north, east) and spacing (m) attached as a header

The encoding is binary to keep the wire compact: a single float32
array plus a small JSON header. The first byte of the WS frame
encodes the schema version so the client can reject mismatches.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Optional


WAVE_PATCH_VERSION = 1
DEFAULT_GRID_SIZE = 65
DEFAULT_EXTENT_M = 128.0
DEFAULT_SPACING_M = 2.0


class WavePatchError(ValueError):
    pass


@dataclass(frozen=True)
class WavePatchSpec:
    """Geometry of a wave patch.

    `size` is the number of samples per side (always square). `extent_m`
    is the side length in metres. The patch is centred on the vehicle's
    horizontal position at sample time.
    """

    size: int = DEFAULT_GRID_SIZE
    extent_m: float = DEFAULT_EXTENT_M

    @property
    def spacing_m(self) -> float:
        if self.size <= 1:
            raise WavePatchError("grid size must be > 1")
        return self.extent_m / (self.size - 1)

    def validate(self) -> None:
        if self.size < 4:
            raise WavePatchError(f"grid size {self.size} too small (<4)")
        if self.size % 2 == 0:
            raise WavePatchError(f"grid size {self.size} must be odd")
        if not (8.0 <= self.extent_m <= 1024.0):
            raise WavePatchError(
                f"extent {self.extent_m} out of [8, 1024] m"
            )

    def to_dict(self) -> dict:
        return {
            "size": int(self.size),
            "extent_m": float(self.extent_m),
            "spacing_m": float(self.spacing_m),
        }


@dataclass(frozen=True)
class WavePatchFrame:
    """A single wave patch sample."""

    sim_time_s: float
    centre_north_m: float
    centre_east_m: float
    spec: WavePatchSpec
    heights: tuple  # size*size float32 row-major, north increasing

    def validate(self) -> None:
        expected = self.spec.size * self.spec.size
        if len(self.heights) != expected:
            raise WavePatchError(
                f"heights length {len(self.heights)} != size^2 {expected}"
            )
        for h in self.heights:
            if not math.isfinite(float(h)):
                raise WavePatchError("non-finite height in patch")


def sample_wave_heights(sea, centre_north_m: float, centre_east_m: float,
                        spec: WavePatchSpec, sim_time_s: float) -> tuple:
    """Sample the ocean at every grid node.

    Prefer a single vectorised eta(x, y, t) call (DirectionalOcean).
    1-D oceans (eta(x, t)) are extruded along east. The grid is centred
    on (centre_north_m, centre_east_m), north increasing along the first
    axis, east increasing along the second.
    """
    spec.validate()
    half = spec.extent_m / 2.0
    step = spec.spacing_m
    n = spec.size
    if np is None:
        return _sample_wave_heights_loop(
            sea, centre_north_m, centre_east_m, spec, sim_time_s,
        )
    n_axis = -half + np.arange(n, dtype=float) * step
    xs = float(centre_north_m) + n_axis
    ys = float(centre_east_m) + n_axis
    grid = None
    eta_fn = getattr(sea, "eta", None)
    if eta_fn is not None:
        try:
            raw = eta_fn(xs, ys, float(sim_time_s))
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == n:
                grid = arr.T
        except TypeError:
            try:
                raw = eta_fn(xs, float(sim_time_s))
                line = np.asarray(raw, dtype=float).reshape(-1)
                if line.size == n:
                    grid = np.repeat(line.reshape(n, 1), n, axis=1)
            except Exception:
                grid = None
        except Exception:
            grid = None
    if grid is None:
        return _sample_wave_heights_loop(
            sea, centre_north_m, centre_east_m, spec, sim_time_s,
        )
    return tuple(float(h) for h in np.asarray(grid, dtype=float).ravel())


def _sample_wave_heights_loop(sea, centre_north_m, centre_east_m,
                              spec, sim_time_s) -> tuple:
    half = spec.extent_m / 2.0
    step = spec.spacing_m
    n = spec.size
    heights = [0.0] * (n * n)
    for i in range(n):
        n_off = -half + i * step
        for j in range(n):
            e_off = -half + j * step
            try:
                eta = sea.eta(
                    np.array([centre_north_m + n_off]),
                    np.array([centre_east_m + e_off]),
                    sim_time_s,
                )
                h = float(eta[0, 0]) if hasattr(eta, "shape") else float(eta)
            except Exception:
                try:
                    eta = sea.eta(
                        centre_north_m + n_off, sim_time_s,
                    )
                    h = float(np.asarray(eta).reshape(-1)[0])
                except Exception:
                    h = 0.0
            heights[i * n + j] = float(h)
    return tuple(heights)


def encode_wave_patch(frame: WavePatchFrame) -> bytes:
    """Encode a WavePatchFrame as a binary blob.

    Layout:
        [0]      : schema version (uint8)
        [1..4]   : sim_time_s (float32 LE)
        [5..8]   : centre_north_m (float32 LE)
        [9..12]  : centre_east_m (float32 LE)
        [13..14] : grid size (uint16 LE)
        [15..18] : spacing_m (float32 LE)
        [19..]   : float32 height grid row-major

    The first 19 bytes form the binary header; clients verify the
    schema version and read the size to size the array.
    """
    frame.validate()
    header = struct.pack(
        "<B f f f H f",
        WAVE_PATCH_VERSION,
        float(frame.sim_time_s),
        float(frame.centre_north_m),
        float(frame.centre_east_m),
        int(frame.spec.size),
        float(frame.spec.spacing_m),
    )
    arr = struct.pack(f"<{len(frame.heights)}f", *frame.heights)
    return header + arr


def encode_wave_patch_message(frame: WavePatchFrame, *, session_id: str = "",
                              epoch: int = 0, max_payload_bytes: int = 256 * 1024
                              ) -> bytes:
    """Encode the wire message: JSON header + '\n' + binary payload."""
    header = json.dumps({
        "v": WAVE_PATCH_VERSION,
        "type": "wave_patch",
        "session_id": session_id,
        "epoch": int(epoch),
        "sim_time_s": float(frame.sim_time_s),
        "centre_north_m": float(frame.centre_north_m),
        "centre_east_m": float(frame.centre_east_m),
        "spec": frame.spec.to_dict(),
    }, allow_nan=False)
    payload = encode_wave_patch(frame)
    blob = (header + "\n").encode("utf-8") + payload
    if len(blob) > max_payload_bytes:
        raise WavePatchError(
            f"wave patch message {len(blob)} exceeds {max_payload_bytes}"
        )
    return blob


def decode_wave_patch_header(prefix: bytes) -> dict:
    """Decode the JSON prefix that comes before the binary payload."""
    text = prefix.decode("utf-8").rstrip("\n")
    obj = json.loads(text)
    if obj.get("v") != WAVE_PATCH_VERSION:
        raise WavePatchError(
            f"unsupported wave patch version: {obj.get('v')!r}"
        )
    return obj


def decode_wave_patch(blob: bytes) -> WavePatchFrame:
    """Decode a single binary payload (no JSON header)."""
    if len(blob) < 19:
        raise WavePatchError(f"blob {len(blob)} < 19 header bytes")
    (version, sim_t, cn, ce, size, spacing) = struct.unpack(
        "<B f f f H f", blob[:19],
    )
    if version != WAVE_PATCH_VERSION:
        raise WavePatchError(f"unknown patch version: {version}")
    expected = size * size
    payload = blob[19:]
    if len(payload) != expected * 4:
        raise WavePatchError(
            f"payload {len(payload)} != {expected}*4 bytes"
        )
    heights = struct.unpack(f"<{expected}f", payload)
    extent = spacing * (size - 1)
    spec = WavePatchSpec(size=int(size), extent_m=float(extent))
    spec.validate()
    return WavePatchFrame(
        sim_time_s=float(sim_t),
        centre_north_m=float(cn),
        centre_east_m=float(ce),
        spec=spec,
        heights=tuple(float(h) for h in heights),
    )


def patch_extent_matches(spec_a: WavePatchSpec, spec_b: WavePatchSpec) -> bool:
    """True iff two specs produce identical wire layouts."""
    return (spec_a.size == spec_b.size
            and abs(spec_a.spacing_m - spec_b.spacing_m) < 1e-6)


# numpy import is deferred so the module can be imported cheaply by tests
# that do not need the ocean sampler.
try:
    import numpy as np  # noqa: F401
except ImportError:
    np = None  # type: ignore
