"""Coordinate transforms between the simulator's NEU frame and a Three.js
scene frame.

Design §8 commits to explicit axes:

    Simulator (north-east-up, NEU)
        +X  : North
        +Y  : East
        +Z  : Up (altitude above mean sea level)

    Three.js scene (right-up-forward, RUF by convention here)
        +X  : East   (right)
        +Y  : Up     (altitude)
        +Z  : South  (forward into screen; -Z = North)

We provide:

    neu_to_ruf(pos_neu, attitude_zyx) -> dict
        pos_neu   : {"x": N, "y": E, "z": U}
        attitude  : {"heading_deg": ψ, "pitch_deg": θ, "bank_deg": φ}

    forward_basis(attitude) -> (right, up, forward) unit vectors in NEU.

All tests assert the sign conventions explicitly so a future refactor
of the dynamics module cannot silently flip an axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


def _deg2rad(x: float) -> float:
    return float(x) * math.pi / 180.0


@dataclass(frozen=True)
class Attitude:
    """ZYX (heading-pitch-bank) Euler angles in degrees."""

    heading_deg: float = 0.0  # ψ, rotation about Z (up), positive = east of north
    pitch_deg: float = 0.0    # θ, rotation about local Y (right), positive = nose up
    bank_deg: float = 0.0     # φ, rotation about body X (forward), positive = right wing down

    def rotation_matrix(self) -> tuple:
        """Return the 3x3 body-to-NEU rotation matrix as nested tuples.

        The sign convention matches spatial_dynamics.py: positive bank
        rolls the right wing down, which tilts the lift vector east
        when heading north (SPATIAL_SYSTEM_PROMPT in ollama_pilot.py).
        The body frame is (X=forward, Y=right, Z=up). R's columns
        are the body axes expressed in NEU; R's rows are the NEU
        axes expressed in body frame.
        """
        ψ = _deg2rad(self.heading_deg)
        θ = _deg2rad(self.pitch_deg)
        φ = _deg2rad(self.bank_deg)
        cψ, sψ = math.cos(ψ), math.sin(ψ)
        cθ, sθ = math.cos(θ), math.sin(θ)
        cφ, sφ = math.cos(φ), math.sin(φ)
        # NEU basis: R = R_z(ψ) · R_y(-θ) · R_x(-φ)
        # R_x(φ) uses the right-wing-down convention:
        #   right axis (body Y) rotates toward -Z_NE, up axis (body Z)
        #   rotates toward +Y_NE when banked right with heading north.
        # Derivation: see comments in coordinates module.
        r00 = cθ * cψ
        r01 = sφ * sθ * cψ - sψ * cφ
        r02 = -cφ * sθ * cψ - sφ * sψ
        r10 = cθ * sψ
        r11 = cψ * cφ + sφ * sθ * sψ
        r12 = sφ * cψ - cφ * sθ * sψ
        r20 = sθ
        r21 = -sφ * cθ
        r22 = cφ * cθ
        return (
            (r00, r01, r02),
            (r10, r11, r12),
            (r20, r21, r22),
        )


def neu_to_ruf(pos_neu: Mapping[str, float],
               attitude: Attitude) -> dict:
    """Convert NEU position + attitude to a RUF scene payload.

    The body forward vector in NEU is (r00, r10, r20). North is +X, so a
    heading of 0° points the body forward along +X_NE (north). The
    Three.js forward axis is -Z_RUF (south into the scene), so we map
    body forward (r00, r10, r20) → (-r20, ?, ?) for the scene's Z.
    Concretely: scene_x = E (pos.y), scene_y = U (pos.z), scene_z = -N
    (negated pos.x).
    """
    north = float(pos_neu.get("x", 0.0))
    east = float(pos_neu.get("y", 0.0))
    up = float(pos_neu.get("z", 0.0))
    scene = {
        "position": {"x": east, "y": up, "z": -north},
        "heading_deg": attitude.heading_deg,
        "pitch_deg": attitude.pitch_deg,
        "bank_deg": attitude.bank_deg,
    }
    rot = attitude.rotation_matrix()
    # Forward in NEU = first column of R: (r00, r10, r20).
    fx_n, fy_e, fz_u = rot[0][0], rot[1][0], rot[2][0]
    # Right in NEU = second column: (r01, r11, r21).
    rx_n, ry_e, rz_u = rot[0][1], rot[1][1], rot[2][1]
    # Up in NEU = third column: (r02, r12, r22).
    ux_n, uy_e, uz_u = rot[0][2], rot[1][2], rot[2][2]
    scene["forward_ruf"] = {"x": fy_e, "y": fz_u, "z": -fx_n}
    scene["right_ruf"] = {"x": ry_e, "y": rz_u, "z": -rx_n}
    scene["up_ruf"] = {"x": uy_e, "y": uz_u, "z": -ux_n}
    return scene


def altitude_above_origin_neu(pos_neu: Mapping[str, float],
                              origin_neu: Mapping[str, float]) -> float:
    """Return the height of pos above origin in metres (Euclidean)."""
    dx = float(pos_neu.get("x", 0.0)) - float(origin_neu.get("x", 0.0))
    dy = float(pos_neu.get("y", 0.0)) - float(origin_neu.get("y", 0.0))
    dz = float(pos_neu.get("z", 0.0)) - float(origin_neu.get("z", 0.0))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def normalise_ruf_axis(vec: Mapping[str, float]) -> tuple:
    """Return (length, (x, y, z)) for a RUF vector dict."""
    x = float(vec.get("x", 0.0))
    y = float(vec.get("y", 0.0))
    z = float(vec.get("z", 0.0))
    length = math.sqrt(x * x + y * y + z * z)
    return length, (x, y, z)


def assert_sign_convention(attitude: Attitude) -> dict:
    """Smoke test for the sign convention.

    With heading=0, pitch=0, bank=0 the body forward should point north
    (+X_NE) and the scene forward should point south (-Z_RUF). With
    bank=+15° (right wing down), the body up vector should lean in the
    +Y_NE (east) direction.
    """
    flat = neu_to_ruf({"x": 0, "y": 0, "z": 100.0},
                      Attitude(heading_deg=0, pitch_deg=0, bank_deg=0))
    # Position: scene.x=0, scene.y=100, scene.z=0
    assert flat["position"]["x"] == 0.0
    assert flat["position"]["y"] == 100.0
    assert flat["position"]["z"] == 0.0
    # Forward in RUF: scene.z < 0 (south into screen).
    assert flat["forward_ruf"]["z"] < 0
    # Right in RUF: scene.x > 0 (east is right).
    assert flat["right_ruf"]["x"] > 0
    # Up in RUF: scene.y > 0.
    assert flat["up_ruf"]["y"] > 0

    banked = neu_to_ruf({"x": 0, "y": 0, "z": 100.0},
                        Attitude(heading_deg=0, pitch_deg=0, bank_deg=15))
    # With bank +15, the body's up should tilt towards scene.x > 0 (east).
    assert banked["up_ruf"]["x"] > 0, banked
    # Forward stays pure north (bank only rolls, doesn't pitch forward).
    assert banked["forward_ruf"]["z"] < 0
    assert abs(banked["forward_ruf"]["y"]) < 1e-6
    # Right wing (east) tilts down: scene.y < 0.
    assert banked["right_ruf"]["y"] < 0, banked
    return {"flat": flat, "banked": banked}
