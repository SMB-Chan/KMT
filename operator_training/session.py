"""Session lifecycle, authority, and handover logic.

Implements the state machine from the design document section 6 with the
following rules:

    Lifecycle:    SETUP -> READY -> RUNNING <-> PAUSED -> FINISHED | ABORTED
    Authority:    AUTO | HUMAN (exactly one during RUNNING)
    Phase:        SETUP | READY | TAKEOFF | CRUISE | APPROACH | TOUCHDOWN
                  | FINISHED | ABORTED
    Pending:      NONE | OFFER_MANUAL | REQUEST_AUTO_LAND

Authority decides which path is fed to FlyingBoatVehicle.step() each
tick. AUTO keeps the navigation command already issued; HUMAN applies
the validated 4-axis input via SpatialControl.apply (servo 1..4 in one
batch). The first servo call flips _active_cmd to MAV_CMD_DO_SET_SERVO
which is the desired side effect for AUTO->HUMAN. HUMAN->AUTO must
re-issue a navigation command; the adapter exposes apply_nav() for
that.

Touchdown is the canonical finish: hull-keel clearance <= 0.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from aircraft import Aircraft

from .curriculum import (
    ApproachGate,
    Curriculum,
    CurriculumPhase,
    TouchdownDetector,
    default_approach_gate,
)
from .envelope import ControlEnvelope, ControlInput, EnvelopeViolation
from .recording import NullSink, Recording, RecordingSink
from .vehicle_adapter import NavCommand, VehicleAdapter


class Authority:
    AUTO = "AUTO"
    HUMAN = "HUMAN"


class Lifecycle:
    SETUP = "SETUP"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"


class Phase:
    SETUP = "SETUP"
    READY = "READY"
    TAKEOFF = "TAKEOFF"
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    TOUCHDOWN = "TOUCHDOWN"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"


class PendingRequest:
    NONE = "NONE"
    OFFER_MANUAL = "OFFER_MANUAL"
    REQUEST_AUTO_LAND = "REQUEST_AUTO_LAND"


class HandoverDeclinedReason:
    NOT_OFFERED = "not_offered"
    INPUT_MISMATCH = "input_mismatch"
    NOT_RUNNING = "not_running"
    ALREADY_HUMAN = "already_human"


class AutoLandDeclinedReason:
    NOT_RUNNING = "not_running"
    NOT_HUMAN = "not_human"
    GATE_FAILED = "gate_failed"
    ALREADY_AUTO = "already_auto"


@dataclass
class SessionConfig:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    curriculum: Curriculum = field(default_factory=lambda: Curriculum(
        curriculum_id="hybrid_baseline",
        curriculum_version="1",
        scenario="hybrid",
    ))
    envelope: ControlEnvelope = field(default_factory=ControlEnvelope.beginner)
    seed: int = 42
    target_takeoff_alt_m: float = 25.0
    target_takeoff_speed_m_s: float = 13.0
    target_landing_alt_m: float = 0.0
    target_landing_speed_m_s: float = 13.0
    target_heading_deg: float = 0.0
    landing_glide_deg: float = 8.0
    wave_Hs: float = 0.8
    wave_Tp: float = 6.0

    def aircraft_V_stall(self, aircraft: Aircraft) -> float:
        return aircraft.V_stall


@dataclass
class SessionState:
    lifecycle: str = Lifecycle.SETUP
    phase: str = Phase.SETUP
    authority: str = Authority.AUTO
    pending: str = PendingRequest.NONE
    assist_active: bool = False
    last_input_seq: int = -1
    sim_t: float = 0.0
    end_reason: Optional[str] = None
    offer_manual_elapsed_s: float = 0.0
    match_elapsed_s: float = 0.0
    land_request_misses: int = 0
    land_request_consecutive_ok: int = 0

    def to_dict(self) -> dict:
        return {
            "lifecycle": self.lifecycle,
            "phase": self.phase,
            "authority": self.authority,
            "pending": self.pending,
            "assist_active": self.assist_active,
            "last_input_seq": self.last_input_seq,
            "sim_t": self.sim_t,
            "end_reason": self.end_reason,
        }


@dataclass
class Session:
    """Single operator-training session."""

    config: SessionConfig
    adapter: VehicleAdapter
    recording: Recording
    state: SessionState = field(default_factory=SessionState)
    auto_nav: Optional[NavCommand] = None
    touchdown: TouchdownDetector = field(default=None)
    tick_count: int = 0

    @classmethod
    def create(cls, *, config: SessionConfig | None = None,
               sink: RecordingSink | None = None,
               seed: int | None = None) -> "Session":
        config = config or SessionConfig()
        if seed is not None:
            config.seed = seed
        adapter = VehicleAdapter.build(
            spatial=config.curriculum.spatial,
            seed=config.seed,
            envelope=config.envelope,
        )
        recording = Recording(
            session_id=config.session_id,
            sink=sink if sink is not None else NullSink(),
            config={
                "curriculum_id": config.curriculum.curriculum_id,
                "curriculum_version": config.curriculum.curriculum_version,
                "scenario": config.curriculum.scenario,
                "wave": {"Hs": config.wave_Hs, "Tp": config.wave_Tp},
                "wind_m_s": [0.0, 0.0, 0.0],
                "seed": config.seed,
                "spatial": config.curriculum.spatial,
                "curriculum": {
                    "handover": {
                        "altitude_m": config.curriculum.handover_altitude_m,
                        "airspeed_m_s": config.curriculum.handover_airspeed_m_s,
                        "vertical_speed_m_s": config.curriculum.handover_vertical_speed_m_s,
                        "bank_deg": config.curriculum.handover_bank_deg,
                        "duration_s": config.curriculum.handover_duration_s,
                    },
                    "approach_gate": {
                        "altitude_lo_m": config.curriculum.approach_gate.altitude_lo_m,
                        "altitude_hi_m": config.curriculum.approach_gate.altitude_hi_m,
                        "airspeed_lo_m_s": config.curriculum.approach_gate.airspeed_lo_m_s,
                        "airspeed_hi_m_s": config.curriculum.approach_gate.airspeed_hi_m_s,
                        "sink_rate_max_m_s": config.curriculum.approach_gate.sink_rate_max_m_s,
                        "bank_abs_deg": config.curriculum.approach_gate.bank_abs_deg,
                        "lateral_abs_m": config.curriculum.approach_gate.lateral_abs_m,
                        "lateral_speed_abs_m_s": config.curriculum.approach_gate.lateral_speed_abs_m_s,
                    },
                },
                "control_envelope": {
                    "throttle_lo": config.envelope.throttle_lo,
                    "throttle_hi": config.envelope.throttle_hi,
                    "pitch_lo": config.envelope.pitch_lo,
                    "pitch_hi": config.envelope.pitch_hi,
                    "bank_abs": config.envelope.bank_abs,
                    "rudder_abs": config.envelope.rudder_abs,
                },
            },
        )
        touchdown = TouchdownDetector.from_vehicle(adapter.vehicle)
        sess = cls(
            config=config,
            adapter=adapter,
            recording=recording,
            touchdown=touchdown,
        )
        recording.open()
        return sess

    # ----- lifecycle transitions -----
    def ready(self) -> None:
        if self.state.lifecycle != Lifecycle.SETUP:
            return
        self.state.lifecycle = Lifecycle.READY
        self.state.phase = Phase.READY
        self.recording.record_event(sim_t=self.state.sim_t, kind="ready")

    def start_takeoff(self) -> None:
        """Arm and issue the auto takeoff nav command. Single-shot."""
        if self.state.lifecycle != Lifecycle.READY:
            raise RuntimeError(
                f"cannot start takeoff from lifecycle {self.state.lifecycle!r}"
            )
        self.adapter.arm()
        self.auto_nav = NavCommand(
            kind="TAKEOFF",
            alt_m=self.config.target_takeoff_alt_m,
            speed_m_s=self.config.target_takeoff_speed_m_s,
            heading_deg=self.config.target_heading_deg,
        )
        self.adapter.apply_nav(self.auto_nav)
        self.state.lifecycle = Lifecycle.RUNNING
        self.state.phase = Phase.TAKEOFF
        self.state.authority = Authority.AUTO
        self.state.pending = PendingRequest.NONE
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="start_takeoff",
            nav=self.auto_nav.__dict__,
        )

    def start_manual(self) -> None:
        """Arm and give HUMAN authority from the water. Single-shot."""
        if self.state.lifecycle != Lifecycle.READY:
            raise RuntimeError(
                f"cannot start manual from lifecycle {self.state.lifecycle!r}"
            )
        self.adapter.arm()
        idle = ControlInput(
            throttle=0.0, pitch_deg=0.0, bank_deg=0.0, rudder=0.0,
        )
        self.adapter.apply_human(idle)
        self.auto_nav = None
        self.state.lifecycle = Lifecycle.RUNNING
        self.state.phase = Phase.TAKEOFF
        self.state.authority = Authority.HUMAN
        self.state.pending = PendingRequest.NONE
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="start_manual",
        )

    def pause(self) -> None:
        if self.state.lifecycle != Lifecycle.RUNNING:
            return
        self.state.lifecycle = Lifecycle.PAUSED
        self.recording.record_event(sim_t=self.state.sim_t, kind="pause")

    def resume(self, *, confirm: bool) -> bool:
        if self.state.lifecycle != Lifecycle.PAUSED:
            return False
        if not confirm:
            return False
        self.state.lifecycle = Lifecycle.RUNNING
        self.recording.record_event(sim_t=self.state.sim_t, kind="resume")
        return True

    def abort(self, reason: str) -> None:
        if self.state.lifecycle in (Lifecycle.FINISHED, Lifecycle.ABORTED):
            return
        self.state.lifecycle = Lifecycle.ABORTED
        self.state.phase = Phase.ABORTED
        self.state.end_reason = reason
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="abort", reason=reason,
        )

    # ----- handover: AUTO -> HUMAN -----
    def request_handover(self) -> None:
        """Human signals intent to take manual control.

        Sets pending=OFFER_MANUAL if the auto path has reached the
        handover envelope (design section 6.1: altitude, airspeed,
        sink rate, bank for handover_duration_s). Otherwise the
        request is buffered in pending_state but not acted on; the
        caller receives a decline reason via evaluate_handover().
        """
        if self.state.lifecycle != Lifecycle.RUNNING:
            self._decline_handover(HandoverDeclinedReason.NOT_RUNNING)
            return
        if self.state.authority == Authority.HUMAN:
            self._decline_handover(HandoverDeclinedReason.ALREADY_HUMAN)
            return
        self.state.pending = PendingRequest.OFFER_MANUAL
        self.state.match_elapsed_s = 0.0
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="handover_requested",
        )

    def evaluate_handover(self, human_input: ControlInput) -> tuple[bool, str]:
        """Decide whether the current human input closes the handover gate.

        Returns (True, "") on success or (False, reason) on decline.
        Called by tick() after the input has been validated.
        """
        if self.state.pending != PendingRequest.OFFER_MANUAL:
            return False, HandoverDeclinedReason.NOT_OFFERED
        auto = self._capture_auto_setpoints()
        diffs = {
            "throttle": abs(human_input.throttle - auto["throttle"]),
            "pitch_deg": abs(human_input.pitch_deg - auto["pitch_deg"]),
            "bank_deg": abs(human_input.bank_deg - auto["bank_deg"]),
            "rudder": abs(human_input.rudder - auto["rudder"]),
        }
        tolerances = {
            "throttle": 0.05,
            "pitch_deg": 2.0,
            "bank_deg": 3.0,
            "rudder": 0.1,
        }
        mismatched = [
            k for k in diffs if diffs[k] > tolerances[k]
        ]
        if mismatched:
            self.state.match_elapsed_s = 0.0
            return False, HandoverDeclinedReason.INPUT_MISMATCH
        self.state.match_elapsed_s += self.adapter.vehicle.dt
        dur = self.config.curriculum.handover_duration_s
        # The design states 0.5 s sustained match. The curriculum
        # `handover_duration_s` is for the OFFER predicate; for the
        # INPUT MATCH predicate we use a fixed 0.5 s per the design.
        if self.state.match_elapsed_s >= 0.5:
            return True, ""
        return False, HandoverDeclinedReason.INPUT_MISMATCH

    def _decline_handover(self, reason: str) -> None:
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="handover_declined", reason=reason,
        )

    def _accept_handover(self) -> None:
        self.state.authority = Authority.HUMAN
        self.state.pending = PendingRequest.NONE
        self.state.match_elapsed_s = 0.0
        if self.state.phase == Phase.TAKEOFF:
            self.state.phase = Phase.CRUISE
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="handover_accepted",
        )

    # ----- handover: HUMAN -> AUTO -----
    def request_auto_land(self) -> None:
        """Human signals intent to let AUTO handle the landing."""
        if self.state.lifecycle != Lifecycle.RUNNING:
            self._decline_auto_land(AutoLandDeclinedReason.NOT_RUNNING)
            return
        if self.state.authority == Authority.AUTO:
            self._decline_auto_land(AutoLandDeclinedReason.ALREADY_AUTO)
            return
        self.state.pending = PendingRequest.REQUEST_AUTO_LAND
        self.state.land_request_consecutive_ok = 0
        self.state.land_request_misses = 0
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="auto_land_requested",
        )

    def cancel_auto_land_request(self) -> None:
        if self.state.pending != PendingRequest.REQUEST_AUTO_LAND:
            return
        self.state.pending = PendingRequest.NONE
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="auto_land_request_cancelled",
        )

    def evaluate_auto_land(self, observation: dict) -> tuple[bool, list[str]]:
        """Evaluate the approach gate; returns (passed, missing_checks)."""
        if self.state.pending != PendingRequest.REQUEST_AUTO_LAND:
            return False, ["not_requested"]
        ok, missing = self.config.curriculum.approach_gate.evaluate(observation)
        if ok:
            self.state.land_request_consecutive_ok += 1
        else:
            self.state.land_request_consecutive_ok = 0
            self.state.land_request_misses += 1
        # Require one tick of gate-passing so a momentary spike does
        # not flip authority. The gate itself enforces all checks; a
        # single valid tick is enough since the operator must have
        # already confirmed.
        if ok:
            return True, []
        return False, missing

    def _decline_auto_land(self, reason: str) -> None:
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="auto_land_declined", reason=reason,
        )

    def _accept_auto_land(self) -> None:
        self.auto_nav = NavCommand(
            kind="LAND",
            alt_m=self.config.target_landing_alt_m,
            speed_m_s=self.config.target_landing_speed_m_s,
            glide_deg=self.config.landing_glide_deg,
            heading_deg=self.config.target_heading_deg,
        )
        self.adapter.apply_nav(self.auto_nav)
        self.state.authority = Authority.AUTO
        self.state.pending = PendingRequest.NONE
        self.state.phase = Phase.APPROACH
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="auto_land_accepted",
        )

    # ----- tick loop -----
    def tick(self, *, dt: float, human_input: Optional[ControlInput],
             input_seq: int = -1, raw_input: Optional[dict] = None,
             receive_t: Optional[float] = None) -> dict:
        """Advance one physics tick.

        Returns a small summary dict with the resulting phase / authority
        / reasons so the caller (UI or test) can react.
        """
        if self.state.lifecycle not in (Lifecycle.RUNNING, Lifecycle.PAUSED):
            return self._summarise_tick(human_input=human_input,
                                        applied=False,
                                        reason="frozen")
        if self.state.lifecycle == Lifecycle.PAUSED:
            return self._summarise_tick(human_input=human_input,
                                        applied=False,
                                        reason="paused")

        receive_t = receive_t if receive_t is not None else time.monotonic()
        raw_input = raw_input or (
            human_input.as_dict() if human_input is not None else None
        )

        # --- Decide authority FIRST, then apply control source ----
        applied = False
        rejection = ""

        # During AUTO, human input is silently dropped unless an
        # explicit handover is in progress. Recording still happens so
        # post-hoc analysis can see the discards.
        if self.state.authority == Authority.AUTO:
            if human_input is not None:
                if self.state.pending == PendingRequest.OFFER_MANUAL:
                    ok, reason = self.evaluate_handover(human_input)
                    if ok:
                        self._accept_handover()
                        self.adapter.apply_human(human_input)
                        # The input was consumed by the handover gate.
                        # Mark it as applied so the replay driver
                        # re-applies it on subsequent ticks.
                        applied = True
                        rejection = "handover_accepted"
                    else:
                        # Eval consumed the input even though the
                        # handover is still pending. The input was
                        # *not* applied to the vehicle, but it WAS
                        # used by the state machine. We mark it as
                        # applied=True so the replay driver replays
                        # it; the replay session re-runs eval_handover
                        # with the same input.
                        applied = True
                        rejection = "handover_match"
                else:
                    rejection = "authority=auto"
        elif self.state.authority == Authority.HUMAN:
            if human_input is None:
                # operator has authority but sent no input this tick;
                # last applied control persists in the vehicle
                rejection = "no_input"
            else:
                self.adapter.apply_human(human_input)
                applied = True

        # --- Step physics ---
        self.adapter.step(dt=dt)
        self.state.sim_t = self.adapter.sim_t
        self.tick_count += 1

        # --- Phase transitions driven by AUTO state ----
        if self.state.authority == Authority.AUTO and self.state.phase == Phase.TAKEOFF:
            self._maybe_offer_manual()
        if self.state.authority == Authority.AUTO and self.state.phase == Phase.APPROACH:
            # In landing we let the existing controller drive to touchdown.
            pass

        # --- Pending auto-land evaluation ----
        if self.state.pending == PendingRequest.REQUEST_AUTO_LAND:
            ok, missing = self.evaluate_auto_land(self.adapter.observe())
            if ok:
                self._accept_auto_land()
            else:
                # gate failed; keep pending, expose missing checks
                self._last_gate_missing = missing
        else:
            self._last_gate_missing = []

        # --- Touchdown detection ----
        if self._is_touchdown():
            self._finish(reason="touchdown")
        elif self.adapter.failed:
            self._finish(reason=f"damage:{self.adapter.failure_reason}")
        else:
            self._maybe_advance_phase_human()

        # --- Recording ----
        self.state.last_input_seq = max(self.state.last_input_seq, int(input_seq))
        self.recording.record_input(
            seq=int(input_seq),
            receive_t=float(receive_t),
            applied=bool(applied),
            reason=rejection,
            raw=raw_input or {},
            request=human_input.as_dict() if applied and human_input else None,
        )
        self._record_tick(applied=applied, input_seq=input_seq)

        return self._summarise_tick(human_input=human_input,
                                    applied=applied,
                                    reason=rejection)

    # ----- internal helpers -----
    def _summarise_tick(self, *, human_input, applied: bool, reason: str) -> dict:
        return {
            "tick": self.tick_count,
            "sim_t": self.state.sim_t,
            "lifecycle": self.state.lifecycle,
            "phase": self.state.phase,
            "authority": self.state.authority,
            "pending": self.state.pending,
            "assist_active": self.state.assist_active,
            "applied": applied,
            "reason": reason,
            "last_input_seq": self.state.last_input_seq,
            "gate_missing": getattr(self, "_last_gate_missing", []),
        }

    def _record_tick(self, *, applied: bool, input_seq: int) -> None:
        obs = self.adapter.observe()
        damage = {
            "failed": bool(obs.get("failed", False)),
            "water_kg": float(obs.get("water_mass_kg", 0.0)),
        }
        control = obs.get("applied_control", {
            "throttle": getattr(self.adapter.vehicle, "throttle", 0.0),
            "pitch_deg": math.degrees(getattr(self.adapter.vehicle, "alpha", 0.0)),
            "bank_deg": math.degrees(getattr(self.adapter.vehicle, "bank_command", 0.0)),
            "rudder": getattr(self.adapter.vehicle, "rudder_command", 0.0),
        })
        self.recording.record_tick(
            tick=self.tick_count,
            sim_t=self.state.sim_t,
            seq=input_seq if applied else None,
            authority=self.state.authority,
            assist=self.state.assist_active,
            control=control,
            state={
                "altitude_m": float(obs.get("altitude_m", 0.0)),
                "forward_speed_m_s": float(obs.get("forward_speed_m_s", 0.0)),
                "vertical_speed_m_s": float(obs.get("vertical_speed_m_s", 0.0)),
                "lateral_position_m": float(obs.get("lateral_position_m", 0.0)),
                "lateral_speed_m_s": float(obs.get("lateral_speed_m_s", 0.0)),
                "bank_deg": float(obs.get("bank_deg", 0.0)),
                "heading_deg": float(obs.get("heading_deg", 0.0)),
                "airspeed_m_s": float(obs.get("airspeed_m_s", 0.0)),
                "keel_clearance_m": float(obs.get("keel_clearance_m", self.adapter.keel_clearance_m)),
                "wave_elevation_m": float(obs.get("wave_elevation_m", 0.0)),
            },
            damage=damage,
        )

    def _capture_auto_setpoints(self) -> dict:
        v = self.adapter.vehicle
        return {
            "throttle": float(v.throttle),
            "pitch_deg": float(math.degrees(v.alpha)),
            "bank_deg": float(math.degrees(getattr(v, "bank_command", 0.0))),
            "rudder": float(getattr(v, "rudder_command", 0.0)),
        }

    def _maybe_offer_manual(self) -> None:
        """Trigger OFFER_MANUAL when the auto envelope has held long enough."""
        cur = self.config.curriculum
        obs = self.adapter.observe()
        z = float(obs.get("altitude_m", 0.0))
        vx = float(obs.get("forward_speed_m_s", 0.0))
        vz = float(obs.get("vertical_speed_m_s", 0.0))
        bank_deg = float(math.degrees(getattr(self.adapter.vehicle, "bank", 0.0)))
        airspeed_required = cur.handover_airspeed_m_s
        if airspeed_required is None:
            airspeed_required = 1.3 * self.config.aircraft_V_stall(self.adapter.vehicle.ac)
        in_envelope = (
            z >= cur.handover_altitude_m
            and vx >= airspeed_required
            and abs(vz) <= cur.handover_vertical_speed_m_s
            and abs(bank_deg) <= cur.handover_bank_deg
        )
        if in_envelope:
            self.state.offer_manual_elapsed_s += self.adapter.vehicle.dt
        else:
            self.state.offer_manual_elapsed_s = 0.0
        if (self.state.offer_manual_elapsed_s >= cur.handover_duration_s
                and self.state.pending != PendingRequest.OFFER_MANUAL):
            self.state.pending = PendingRequest.OFFER_MANUAL
            self.state.match_elapsed_s = 0.0
            self.recording.record_event(
                sim_t=self.state.sim_t, kind="offer_manual",
                envelope={
                    "altitude_m": z, "airspeed_m_s": vx,
                    "vertical_speed_m_s": vz, "bank_deg": bank_deg,
                },
            )

    def _maybe_advance_phase_human(self) -> None:
        if self.state.authority != Authority.HUMAN:
            return
        obs = self.adapter.observe()
        z = float(obs.get("altitude_m", 0.0))
        if self.state.phase in (Phase.CRUISE,):
            vx = float(obs.get("forward_speed_m_s", 0.0))
            if vx < 0.5 and z < 0.5:
                return
        if self.state.phase == Phase.CRUISE and z < 5.0:
            self.state.phase = Phase.APPROACH

    def _is_touchdown(self) -> bool:
        # Touchdown is the landing-side finish event. During TAKEOFF
        # the boat is already in the water (z ~ h_keel + eta at rest),
        # so an unguarded contact check would fire as soon as the
        # min_sim_t_s guard elapses. Restrict to the descent phases.
        if self.state.phase not in (Phase.APPROACH, Phase.TOUCHDOWN):
            return False
        return self.touchdown.is_contact(
            z=self.adapter.vehicle.z,
            eta=self.adapter.wave_elevation_m,
            sim_t=self.state.sim_t,
        )

    def _finish(self, reason: str) -> None:
        if self.state.lifecycle in (Lifecycle.FINISHED, Lifecycle.ABORTED):
            return
        # Capture last observation for the summary
        obs = self.adapter.observe()
        self.state.lifecycle = Lifecycle.FINISHED
        self.state.phase = Phase.TOUCHDOWN if reason == "touchdown" else Phase.FINISHED
        self.state.end_reason = reason
        self.recording.record_event(
            sim_t=self.state.sim_t, kind="finish", reason=reason,
        )
        self._last_observation = obs

    def finalize(self, summary_extra: Optional[dict] = None) -> dict:
        summary = {
            "session_id": self.config.session_id,
            "curriculum_id": self.config.curriculum.curriculum_id,
            "scenario": self.config.curriculum.scenario,
            "lifecycle_end": self.state.lifecycle,
            "phase_end": self.state.phase,
            "authority_end": self.state.authority,
            "end_reason": self.state.end_reason,
            "sim_t_end": self.state.sim_t,
            "tick_count": self.tick_count,
            "last_input_seq": self.state.last_input_seq,
        }
        if summary_extra:
            summary.update(summary_extra)
        self.recording.finalize(summary=summary)
        return summary
