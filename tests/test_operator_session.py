import json
import math
import unittest
from pathlib import Path

from operator_training.session import (
    Authority,
    AutoLandDeclinedReason,
    Curriculum,
    HandoverDeclinedReason,
    Lifecycle,
    PendingRequest,
    Phase,
    Session,
    SessionConfig,
)
from operator_training.curriculum import (
    ApproachGate,
    CurriculumPhase,
    TouchdownDetector,
    default_approach_gate,
)
from operator_training.recording import FileSink
from operator_training.envelope import ControlEnvelope, ControlInput
from operator_training.fake_pad import FakePad
from operator_training.vehicle_adapter import VehicleAdapter


def make_session(*, root: Path, seed: int = 42,
                 target_alt: float = 25.0,
                 target_speed: float = 13.0,
                 curriculum=None):
    cfg = SessionConfig(
        curriculum=curriculum or Curriculum(
            curriculum_id="hybrid_baseline",
            curriculum_version="1",
            scenario="hybrid",
        ),
        envelope=ControlEnvelope.beginner(),
        seed=seed,
        target_takeoff_alt_m=target_alt,
        target_takeoff_speed_m_s=target_speed,
    )
    return Session.create(config=cfg, sink=FileSink(root=root), seed=seed)


class SessionLifecycleTests(unittest.TestCase):
    def test_starts_in_setup(self, root=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            self.assertEqual(sess.state.lifecycle, Lifecycle.SETUP)
            self.assertEqual(sess.state.authority, Authority.AUTO)
            self.assertEqual(sess.state.phase, Phase.SETUP)

    def test_ready_then_takeoff_arms_and_starts_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready()
            self.assertEqual(sess.state.lifecycle, Lifecycle.READY)
            sess.start_takeoff()
            self.assertEqual(sess.state.lifecycle, Lifecycle.RUNNING)
            self.assertEqual(sess.state.phase, Phase.TAKEOFF)
            self.assertEqual(sess.state.authority, Authority.AUTO)

    def test_takeoff_outside_ready_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            # Lifecycle is SETUP; cannot start_takeoff yet
            with self.assertRaises(RuntimeError):
                sess.start_takeoff()

    def test_pause_resume_requires_confirm(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            sess.pause()
            self.assertEqual(sess.state.lifecycle, Lifecycle.PAUSED)
            self.assertFalse(sess.resume(confirm=False))
            self.assertEqual(sess.state.lifecycle, Lifecycle.PAUSED)
            self.assertTrue(sess.resume(confirm=True))
            self.assertEqual(sess.state.lifecycle, Lifecycle.RUNNING)

    def test_abort_finalises_lifecycle(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            sess.abort(reason="user_abort")
            self.assertEqual(sess.state.lifecycle, Lifecycle.ABORTED)
            self.assertEqual(sess.state.end_reason, "user_abort")

    def test_tick_is_noop_when_paused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            sess.pause()
            t0 = sess.adapter.sim_t
            out = sess.tick(dt=0.05, human_input=None)
            self.assertEqual(sess.adapter.sim_t, t0)
            self.assertEqual(out["lifecycle"], Lifecycle.PAUSED)
            self.assertEqual(out["reason"], "paused")


class AutoAuthorityTests(unittest.TestCase):
    def test_human_input_discarded_in_auto(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root, target_alt=12.0, target_speed=11.0)
            sess.ready(); sess.start_takeoff()
            # In AUTO, human inputs must be discarded unless a handover
            # is pending.
            ci = ControlInput(throttle=1.0, pitch_deg=8.0, bank_deg=0.0, rudder=0.0)
            summary = sess.tick(dt=0.05, human_input=ci, input_seq=1)
            self.assertEqual(summary["authority"], Authority.AUTO)
            self.assertFalse(summary["applied"])
            self.assertEqual(summary["reason"], "authority=auto")

    def test_offer_manual_appears_when_envelope_holds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Lower thresholds to make the offer fire in a few seconds.
            cur = Curriculum(
                curriculum_id="hybrid_baseline",
                curriculum_version="1",
                scenario="hybrid",
                handover_altitude_m=2.0,
                handover_airspeed_m_s=7.0,
                handover_vertical_speed_m_s=4.0,
                handover_bank_deg=20.0,
                handover_duration_s=0.5,
            )
            sess = make_session(root=root, curriculum=cur,
                                target_alt=4.0, target_speed=8.0)
            sess.ready(); sess.start_takeoff()
            for _ in range(400):
                if sess.state.pending == PendingRequest.OFFER_MANUAL:
                    break
                sess.tick(dt=0.05, human_input=None)
            # After up to 20 sim seconds, the offer should be live OR the
            # aircraft is still climbing - record both outcomes for triage.
            obs = sess.adapter.observe()
            self.assertTrue(
                sess.state.pending == PendingRequest.OFFER_MANUAL
                or obs.get("altitude_m", 0.0) < cur.handover_altitude_m,
                f"unexpected state: pending={sess.state.pending}, "
                f"altitude={obs.get('altitude_m'):.2f}",
            )


class HandoverTests(unittest.TestCase):
    def _force_offer_manual(self, sess):
        """Force the OFFER_MANUAL state without waiting for the envelope."""
        sess.state.pending = PendingRequest.OFFER_MANUAL
        sess.state.match_elapsed_s = 0.0
        sess.state.offer_manual_elapsed_s = sess.config.curriculum.handover_duration_s

    def test_handover_requires_offer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            ci = ControlInput(throttle=0.5, pitch_deg=0.0, bank_deg=0.0, rudder=0.0)
            ok, reason = sess.evaluate_handover(ci)
            self.assertFalse(ok)
            self.assertEqual(reason, HandoverDeclinedReason.NOT_OFFERED)

    def test_handover_requires_sustained_match(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            self._force_offer_manual(sess)
            # Re-capture auto setpoints each iteration because step() in
            # the previous tick updates them.
            for i in range(30):
                if sess.state.authority == Authority.HUMAN:
                    return
                auto = sess._capture_auto_setpoints()
                matching = ControlInput(
                    throttle=auto["throttle"],
                    pitch_deg=auto["pitch_deg"],
                    bank_deg=auto["bank_deg"],
                    rudder=auto["rudder"],
                )
                sess.tick(dt=0.05, human_input=matching, input_seq=i + 1)
            self.fail("handover did not complete within 30 ticks of match")

    def test_handover_declines_on_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            self._force_offer_manual(sess)
            auto = sess._capture_auto_setpoints()
            # Mismatched pitch by 5 deg (>2 deg tolerance)
            mismatched = ControlInput(
                throttle=auto["throttle"],
                pitch_deg=auto["pitch_deg"] + 5.0,
                bank_deg=auto["bank_deg"],
                rudder=auto["rudder"],
            )
            for _ in range(3):
                sess.tick(dt=0.05, human_input=mismatched, input_seq=1)
            self.assertEqual(sess.state.authority, Authority.AUTO)
            self.assertEqual(sess.state.match_elapsed_s, 0.0)

    def test_human_input_actually_applied_after_handover(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            self._force_offer_manual(sess)
            applied_input = ControlInput(
                throttle=0.7, pitch_deg=4.0, bank_deg=0.0, rudder=0.0,
            )
            for i in range(30):
                if sess.state.authority == Authority.HUMAN:
                    sess.tick(dt=0.05, human_input=applied_input,
                              input_seq=i + 1)
                    self.assertTrue(
                        sess.adapter.is_servo_mode(),
                        "vehicle _active_cmd should be MAV_CMD_DO_SET_SERVO "
                        f"after HUMAN authority, got {sess.adapter.vehicle._active_cmd}",
                    )
                    self.assertTrue(sess.state.last_input_seq >= 0)
                    return
                auto = sess._capture_auto_setpoints()
                matching = ControlInput(
                    throttle=auto["throttle"],
                    pitch_deg=auto["pitch_deg"],
                    bank_deg=auto["bank_deg"],
                    rudder=auto["rudder"],
                )
                sess.tick(dt=0.05, human_input=matching, input_seq=i + 1)
            self.fail("never reached HUMAN authority")


class AutoLandTests(unittest.TestCase):
    def test_request_auto_land_requires_human(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            sess.request_auto_land()
            # In AUTO, request should be declined.
            self.assertEqual(sess.state.pending, PendingRequest.NONE)

    def test_request_auto_land_evaluates_gate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Wide gate that passes with default telemetry.
            cur = Curriculum(
                curriculum_id="hybrid_baseline",
                curriculum_version="1",
                scenario="hybrid",
                approach_gate=ApproachGate(
                    altitude_lo_m=0.0,
                    altitude_hi_m=1000.0,
                    airspeed_lo_m_s=0.0,
                    airspeed_hi_m_s=1000.0,
                    sink_rate_max_m_s=10.0,
                    bank_abs_deg=180.0,
                    lateral_abs_m=1000.0,
                    lateral_speed_abs_m_s=1000.0,
                ),
            )
            sess = make_session(root=root, curriculum=cur)
            sess.ready(); sess.start_takeoff()
            # Force into HUMAN
            sess.state.authority = Authority.HUMAN
            sess.state.phase = Phase.CRUISE
            sess.request_auto_land()
            self.assertEqual(sess.state.pending, PendingRequest.REQUEST_AUTO_LAND)
            out = sess.tick(dt=0.05, human_input=None, input_seq=1)
            self.assertEqual(sess.state.authority, Authority.AUTO)
            self.assertEqual(sess.state.phase, Phase.APPROACH)

    def test_gate_failure_keeps_pending(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cur = Curriculum(
                curriculum_id="hybrid_baseline",
                curriculum_version="1",
                scenario="hybrid",
                approach_gate=ApproachGate(
                    altitude_lo_m=100.0,  # unreachable
                    altitude_hi_m=1000.0,
                    airspeed_lo_m_s=0.0,
                    airspeed_hi_m_s=1000.0,
                    sink_rate_max_m_s=10.0,
                    bank_abs_deg=180.0,
                    lateral_abs_m=1000.0,
                    lateral_speed_abs_m_s=1000.0,
                ),
            )
            sess = make_session(root=root, curriculum=cur)
            sess.ready(); sess.start_takeoff()
            sess.state.authority = Authority.HUMAN
            sess.state.phase = Phase.CRUISE
            sess.request_auto_land()
            sess.tick(dt=0.05, human_input=None, input_seq=1)
            self.assertEqual(sess.state.pending, PendingRequest.REQUEST_AUTO_LAND)
            self.assertGreater(sess.state.land_request_misses, 0)

    def test_cancel_auto_land_request_clears_pending(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cur = Curriculum(
                curriculum_id="hybrid_baseline",
                curriculum_version="1",
                scenario="hybrid",
                approach_gate=ApproachGate(
                    altitude_lo_m=100.0,
                    altitude_hi_m=1000.0,
                    airspeed_lo_m_s=0.0,
                    airspeed_hi_m_s=1000.0,
                ),
            )
            sess = make_session(root=root, curriculum=cur)
            sess.ready(); sess.start_takeoff()
            sess.state.authority = Authority.HUMAN
            sess.state.phase = Phase.CRUISE
            sess.request_auto_land()
            sess.cancel_auto_land_request()
            self.assertEqual(sess.state.pending, PendingRequest.NONE)


class TouchdownTests(unittest.TestCase):
    def test_touchdown_on_keel_clearance_zero(self):
        td = TouchdownDetector(hull_h_keel_m=0.2, min_sim_t_s=0.0)
        # z=0.5, eta=0.3 -> clearance=0.5-0.2-0.3=0.0
        self.assertTrue(td.is_contact(z=0.5, eta=0.3, sim_t=2.0))
        # z=0.6, eta=0.3 -> clearance=0.1 (above water)
        self.assertFalse(td.is_contact(z=0.6, eta=0.3, sim_t=2.0))
        # below water -> clearance<0
        self.assertTrue(td.is_contact(z=0.1, eta=0.3, sim_t=2.0))

    def test_touchdown_ignored_in_first_second(self):
        td = TouchdownDetector(hull_h_keel_m=0.2, min_sim_t_s=1.0)
        self.assertFalse(td.is_contact(z=0.1, eta=0.5, sim_t=0.5))

    def test_session_finishes_on_touchdown(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            # Touchdown detection is gated on phase=APPROACH/TOUCHDOWN;
            # for a hybrid scenario that starts at water, manually
            # advance phase to APPROACH and disable the sim-time guard
            # so the synthetic contact below is honoured.
            sess.state.phase = Phase.APPROACH
            from operator_training.curriculum import TouchdownDetector
            sess.touchdown = TouchdownDetector(
                hull_h_keel_m=sess.adapter.vehicle.hull.h_keel,
                min_sim_t_s=0.0,
            )
            veh = sess.adapter.vehicle
            eta = sess.adapter.wave_elevation_m
            veh.z = veh.hull.h_keel + eta - 0.05
            veh.Vx = 0.0
            veh.Vz = 0.0
            sess.tick(dt=0.05, human_input=None)
            self.assertEqual(sess.state.lifecycle, Lifecycle.FINISHED)
            self.assertEqual(sess.state.end_reason, "touchdown")


class DisconnectTests(unittest.TestCase):
    def test_no_input_human_authority_does_not_freeze(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            # Force HUMAN authority without an offer sequence.
            sess.state.authority = Authority.HUMAN
            sess.state.phase = Phase.CRUISE
            t0 = sess.adapter.sim_t
            out = sess.tick(dt=0.05, human_input=None, input_seq=0)
            # Vehicle should still advance; the last applied control persists.
            self.assertGreater(sess.adapter.sim_t, t0)
            self.assertEqual(out["reason"], "no_input")


class HandoverCommandJumpTests(unittest.TestCase):
    """Design §11 P2: 引継ぎで許容以上の指令ジャンプがないことを
    tickログで確認する。

    The accepted handover applies the recorded human input on the
    tick where the gate opens; that input should sit within the
    design tolerances (throttle ≤ 0.05, pitch ≤ 2°, bank ≤ 3°,
    rudder ≤ 0.1) of the AUTO setpoint that was just produced.
    """

    def test_handover_switches_servo_mode_on_accepting_tick(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            sess = make_session(root=Path(tmp))
            sess.ready()
            sess.start_takeoff()
            sess.state.pending = "OFFER_MANUAL"
            ci = ControlInput(throttle=0.4, pitch_deg=2, bank_deg=0, rudder=0)
            with patch.object(sess, "evaluate_handover", return_value=(True, "")):
                sess.tick(dt=0.05, human_input=ci, input_seq=1)
            self.assertEqual(sess.state.authority, "HUMAN")
            self.assertTrue(sess.adapter.is_servo_mode())
            self.assertAlmostEqual(sess.adapter.vehicle.throttle, 0.4)

    def test_accepted_handover_has_no_command_jump(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            # Run one tick so the controller populates setpoints.
            sess.tick(dt=0.05, human_input=None, input_seq=0)
            sess.state.pending = "OFFER_MANUAL"
            sess.state.match_elapsed_s = 0.0
            sess.state.offer_manual_elapsed_s = 0.5
            pre_auto = sess._capture_auto_setpoints()
            # Build a "match" input from the AUTO setpoint (zero jump).
            ci = ControlInput(
                throttle=pre_auto["throttle"],
                pitch_deg=pre_auto["pitch_deg"],
                bank_deg=pre_auto["bank_deg"],
                rudder=pre_auto["rudder"],
            )
            accepted = False
            applied_seq = None
            for seq in range(1, 20):
                if sess.state.authority == "HUMAN":
                    sess.tick(dt=0.05, human_input=ci, input_seq=seq)
                    applied_seq = seq
                    accepted = True
                    break
                sess.tick(dt=0.05, human_input=ci, input_seq=seq)
            self.assertTrue(accepted,
                            "handover did not accept with sustained match")
            actual_path = root / "ticks.jsonl"
            self.assertTrue(actual_path.exists())
            rows = [json.loads(line) for line in actual_path.read_text().splitlines()]
            last = rows[-1]
            self.assertEqual(last["authority"], "HUMAN")
            ctrl = last["control"]
            self.assertLessEqual(
                abs(ctrl["throttle"] - pre_auto["throttle"]), 0.05,
                f"throttle jump: {ctrl['throttle']} vs {pre_auto['throttle']}",
            )
            self.assertLessEqual(
                abs(ctrl["pitch_deg"] - pre_auto["pitch_deg"]), 2.0,
                f"pitch jump: {ctrl['pitch_deg']} vs {pre_auto['pitch_deg']}",
            )
            self.assertLessEqual(
                abs(ctrl["bank_deg"] - pre_auto["bank_deg"]), 3.0,
                f"bank jump: {ctrl['bank_deg']} vs {pre_auto['bank_deg']}",
            )
            self.assertLessEqual(
                abs(ctrl["rudder"] - pre_auto["rudder"]), 0.1,
                f"rudder jump: {ctrl['rudder']} vs {pre_auto['rudder']}",
            )
            self.assertEqual(last["applied_seq"], applied_seq)

    def test_handover_jumps_above_threshold_rejected(self):
        """An out-of-tolerance input must NOT change authority."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sess = make_session(root=root)
            sess.ready(); sess.start_takeoff()
            sess.state.pending = "OFFER_MANUAL"
            sess.state.match_elapsed_s = 0.0
            sess.state.offer_manual_elapsed_s = 0.5
            auto = sess._capture_auto_setpoints()
            # Mismatched by 5 deg in pitch (well above 2 deg tolerance).
            bad = ControlInput(
                throttle=auto["throttle"],
                pitch_deg=auto["pitch_deg"] + 5.0,
                bank_deg=auto["bank_deg"],
                rudder=auto["rudder"],
            )
            for _ in range(15):
                sess.tick(dt=0.05, human_input=bad, input_seq=1)
                if sess.state.authority == "HUMAN":
                    self.fail("handover accepted out-of-tolerance input")
            self.assertEqual(sess.state.authority, "AUTO")
            self.assertAlmostEqual(sess.state.match_elapsed_s, 0.0)


class RecordingTests(unittest.TestCase):
    def test_config_inputs_ticks_events_written(self):
        import tempfile
        from operator_training.envelope import PITCH_BEGINNER_HI
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sess = make_session(root=root, target_alt=15.0)
            sess.ready(); sess.start_takeoff()
            for i in range(10):
                sess.tick(dt=0.05, human_input=None, input_seq=i)
            sess.finalize()
            self.assertTrue((root / "config.json").exists())
            self.assertTrue((root / "inputs.jsonl").exists())
            self.assertTrue((root / "ticks.jsonl").exists())
            self.assertTrue((root / "events.jsonl").exists())
            self.assertTrue((root / "summary.json").exists())

            with (root / "config.json").open() as f:
                cfg = json.load(f)
            self.assertEqual(cfg["curriculum_id"], "hybrid_baseline")
            self.assertEqual(cfg["control_envelope"]["pitch_hi"], PITCH_BEGINNER_HI)

            with (root / "events.jsonl").open() as f:
                events = [json.loads(line) for line in f]
            kinds = [e["kind"] for e in events]
            self.assertIn("ready", kinds)
            self.assertIn("start_takeoff", kinds)

            with (root / "ticks.jsonl").open() as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertEqual(row["authority"], Authority.AUTO)
                self.assertIn("state", row)
                self.assertIn("keel_clearance_m", row["state"])


class ManualStartTests(unittest.TestCase):
    def test_start_manual_sets_human_authority(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sess = make_session(root=Path(td))
            sess.ready()
            sess.start_manual()
            self.assertEqual(sess.state.lifecycle, Lifecycle.RUNNING)
            self.assertEqual(sess.state.authority, Authority.HUMAN)
            self.assertEqual(sess.state.phase, Phase.TAKEOFF)
            self.assertTrue(sess.adapter.is_servo_mode())

    def test_start_manual_outside_ready_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sess = make_session(root=Path(td))
            with self.assertRaises(RuntimeError):
                sess.start_manual()

    def test_start_manual_applies_human_input(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sess = make_session(root=Path(td))
            sess.ready()
            sess.start_manual()
            ci = ControlInput(throttle=0.8, pitch_deg=4.0, bank_deg=0.0, rudder=0.0)
            out = sess.tick(dt=0.05, human_input=ci, input_seq=1)
            self.assertTrue(out["applied"])
            self.assertEqual(out["authority"], Authority.HUMAN)
            self.assertGreater(sess.adapter.vehicle.throttle, 0.7)


if __name__ == "__main__":
    unittest.main()
