import json
import shutil
import tempfile
import unittest
from pathlib import Path

from operator_training import (
    ControlEnvelope,
    Curriculum,
    Session,
    SessionConfig,
)
from operator_training.replay import (
    DEFAULT_TOLERANCES,
    ReplayReport,
    load_config,
    replay_session,
)
from operator_training.recording import FileSink


def _run_smoke(root: Path) -> None:
    cfg = SessionConfig(
        curriculum=Curriculum(
            curriculum_id="hybrid_baseline",
            curriculum_version="1",
            scenario="hybrid",
            handover_altitude_m=4.0,
            handover_airspeed_m_s=8.0,
            handover_vertical_speed_m_s=4.0,
            handover_bank_deg=20.0,
            handover_duration_s=0.5,
        ),
        envelope=ControlEnvelope.beginner(),
        seed=42,
        target_takeoff_alt_m=15.0,
    )
    sess = Session.create(config=cfg, sink=FileSink(root=root))
    sess.ready(); sess.start_takeoff()
    requested = False
    for i in range(200):
        sess.tick(dt=0.05, human_input=None, input_seq=i)
        if sess.state.lifecycle in ("FINISHED", "ABORTED"):
            break
        if not requested and sess.state.pending == "OFFER_MANUAL":
            sess.request_handover()
            requested = True
        if requested:
            auto = sess._capture_auto_setpoints()
            from operator_training import ControlInput
            ci = ControlInput(auto["throttle"], auto["pitch_deg"],
                              auto["bank_deg"], auto["rudder"])
            sess.tick(dt=0.05, human_input=ci, input_seq=i + 1000)
            if sess.state.authority == "HUMAN":
                # Run a few more ticks in HUMAN
                for j in range(10):
                    sess.tick(dt=0.05, human_input=ci, input_seq=i + 2000 + j)
                break
    sess.finalize()


class ReplayDeterminismTests(unittest.TestCase):
    def test_replay_same_seed_matches_original(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            tgt = root / "tgt"
            _run_smoke(src)
            # Now replay into tgt.
            _, report = replay_session(source_root=src, target_root=tgt)
            # The replay should at least compare all ticks and have no
            # mismatches because we used the same seed and same inputs.
            self.assertGreater(report.ticks_compared, 0,
                               f"no ticks compared: {report.to_dict()}")
            if not report.passed:
                # Print mismatches for debugging
                print(json.dumps(report.to_dict(), indent=2))
            self.assertTrue(report.passed, f"replay mismatches: {report.mismatches}")

    def test_load_config_round_trips(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _run_smoke(root)
            cfg = load_config(root)
            self.assertEqual(cfg["curriculum_id"], "hybrid_baseline")
            self.assertEqual(cfg["seed"], 42)
            self.assertIn("control_envelope", cfg)

    def test_replay_with_different_seed_differs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            tgt = root / "tgt"
            _run_smoke(src)
            # Re-run the same scenario with seed=43 to verify the
            # simulator is not bit-equal between seeds. We only step
            # one tick (no inputs) and compare altitude.
            cfg = load_config(src)
            from operator_training import SessionConfig, Session
            from operator_training.recording import FileSink
            from operator_training.envelope import ControlEnvelope
            from operator_training.curriculum import Curriculum
            envelope_dict = cfg["control_envelope"]
            envelope = ControlEnvelope(
                throttle_lo=float(envelope_dict["throttle_lo"]),
                throttle_hi=float(envelope_dict["throttle_hi"]),
                pitch_lo=float(envelope_dict["pitch_lo"]),
                pitch_hi=float(envelope_dict["pitch_hi"]),
                bank_abs=float(envelope_dict["bank_abs"]),
                rudder_abs=float(envelope_dict["rudder_abs"]),
            )
            curriculum = Curriculum(
                curriculum_id=cfg["curriculum_id"],
                curriculum_version=cfg["curriculum_version"],
                scenario=cfg["scenario"],
            )

            def step_one_with_seed(seed: int, root: Path):
                root.mkdir(parents=True, exist_ok=True)
                sess_cfg = SessionConfig(
                    session_id=f"replay_{seed}",
                    curriculum=curriculum,
                    envelope=envelope,
                    seed=seed,
                )
                sess = Session.create(
                    config=sess_cfg, sink=FileSink(root=root), seed=seed,
                )
                sess.ready(); sess.start_takeoff()
                sess.tick(dt=0.05, human_input=None, input_seq=0)
                sess.finalize()

            step_one_with_seed(42, tgt / "s42")
            step_one_with_seed(43, tgt / "s43")
            s42 = json.loads((tgt / "s42" / "ticks.jsonl").read_text().splitlines()[0])
            s43 = json.loads((tgt / "s43" / "ticks.jsonl").read_text().splitlines()[0])
            # At least one state field should differ between seeds.
            differing = []
            for key in ("altitude_m", "forward_speed_m_s",
                        "wave_elevation_m", "keel_clearance_m"):
                a = s42["state"].get(key)
                b = s43["state"].get(key)
                if a is None or b is None:
                    continue
                if abs(a - b) > 1e-9:
                    differing.append(key)
            self.assertGreater(len(differing), 0,
                               "different seeds produced identical state")


class TolerancesTests(unittest.TestCase):
    def test_default_tolerances_are_tight(self):
        # Bit-equal replay should satisfy default tolerances.
        for key, tol in DEFAULT_TOLERANCES.items():
            with self.subTest(key=key):
                self.assertLessEqual(tol, 1e-5)


if __name__ == "__main__":
    unittest.main()
