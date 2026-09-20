"""Tests for design_optimize.py (parametric airframe design study).

Guards:
  * baseline design variables reproduce aircraft.Aircraft() exactly,
  * the own water-run integrator is bitwise-consistent with
    dynamics.simulate_takeoff at the baseline design,
  * baseline evaluation is feasible and pins the cost normalisation,
  * the deterministic pattern search converges on a convex dummy,
  * the end-to-end study writes all artifacts (micro budget, tmp dir).
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import design_optimize as do
from aircraft import Aircraft
from dynamics import simulate_takeoff
from ocean import Ocean

# Micro-budget config so the end-to-end test stays fast.
MICRO = do.StudyConfig(
    takeoff_duration=12.0, takeoff_dt=0.05,
    landing_duration=32.0, landing_dt=0.02,
    final_landing_duration=32.0, final_landing_dt=0.02,
    initial_step_frac=0.5, min_step_frac=0.25, max_rounds=2,
    plots=False)


class ApplyDesignTests(unittest.TestCase):
    def test_baseline_reproduces_aircraft_exactly(self):
        self.assertEqual(do.apply_design(do.BASELINE_DV), Aircraft())

    def test_clip_respects_bounds(self):
        wild = {name: 1e6 for name in do.VAR_ORDER}
        clipped = do.clip_dv(wild)
        for name, spec in do.DESIGN_VARS.items():
            self.assertEqual(clipped[name], spec[2])
        wild = {name: -1e6 for name in do.VAR_ORDER}
        clipped = do.clip_dv(wild)
        for name, spec in do.DESIGN_VARS.items():
            self.assertEqual(clipped[name], spec[1])

    def test_corners_keep_positive_mass_and_margin(self):
        for corner in (1, 2):
            dv = {name: do.DESIGN_VARS[name][corner] for name in do.VAR_ORDER}
            ac = do.apply_design(dv)
            self.assertGreater(ac.mass.total, 0.0)
            self.assertGreaterEqual(ac.mass.margin, 0.4 - 1e-9)
            self.assertTrue(math.isfinite(ac.V_stall))

    def test_eta_closure_matches_ocean_bitwise(self):
        sea = Ocean(Hs=1.5, Tp=6.0, seed=42)
        eta = do.make_eta(sea)
        for x, t in ((0.0, 0.0), (3.7, 1.9), (120.0, 25.0)):
            self.assertEqual(eta(x, t), float(sea.eta(np.array([x]), t)[0]))


class WaterRunParityTests(unittest.TestCase):
    def test_takeoff_matches_simulate_takeoff_at_baseline(self):
        cfg = do.StudyConfig()
        ac = Aircraft()
        sea = Ocean(Hs=1.5, Tp=6.0, seed=42)
        mine = do.run_takeoff(ac, sea, cfg)
        ref = simulate_takeoff(ac, sea, duration=cfg.takeoff_duration,
                               dt=cfg.takeoff_dt)
        # state trajectories bitwise identical
        np.testing.assert_array_equal(mine["Vx"], ref.Vx)
        np.testing.assert_array_equal(mine["x"], ref.x)
        np.testing.assert_array_equal(mine["z"], ref.z)
        np.testing.assert_array_equal(mine["phase"], ref.phase.astype(int))
        # hull resistance: ref logs R = R_hull + D (index 0 differs by design)
        r_ref = ref.R - ref.D
        self.assertLess(np.abs(mine["R_hull"][1:] - r_ref[1:]).max(), 1e-8)
        # liftoff time: mine requires 0.5 s sustained airborne, ref's
        # argmax catches the first momentary phase flip -- compare the
        # sustained index on both sides (phase arrays are bitwise equal)
        dt = cfg.takeoff_dt
        hold = max(1, int(round(do.LIFTOFF_HOLD_S / dt)))
        ph = ref.phase.astype(int)
        lift_ref = -1
        for j in range(0, len(ph) - hold + 1):
            if ph[j] == 1 and ph[j:j + hold].min() == 1:
                lift_ref = j
                break
        self.assertGreaterEqual(lift_ref, 0)
        self.assertAlmostEqual(mine["t_liftoff"], float(ref.t[lift_ref]),
                               delta=dt)


class EvaluationTests(unittest.TestCase):
    def test_baseline_is_feasible_and_normalises_to_one(self):
        cfg = do.StudyConfig()
        ev = do.Evaluator(cfg)
        r = ev.evaluate(do.BASELINE_DV)
        m = r.metrics
        self.assertTrue(m["liftoff"])
        self.assertTrue(m["touchdown"])
        self.assertFalse(m["diverged"])
        # only the landing-load limit may be active at the baseline
        self.assertLessEqual(set(r.violations), {"N_land_peak"})
        # J(baseline) = sum(weights) + landing-load penalty
        self.assertAlmostEqual(r.cost, 1.0, delta=0.2)
        for key in do.WEIGHTS:
            self.assertTrue(math.isfinite(m[key]))
        self.assertGreater(m["t_liftoff"], 0.0)
        self.assertLess(m["t_liftoff"], cfg.takeoff_duration)

    def test_worse_design_costs_more(self):
        cfg = do.StudyConfig(quick=True)
        ev = do.Evaluator(cfg)
        base = ev.evaluate(do.BASELINE_DV).cost
        worse = dict(do.BASELINE_DV)
        worse["margin_kg"] = 6.0      # +6 kg
        worse["CD0"] = 0.030          # dirtier surface
        worse["D_prop"] = 0.61        # smaller prop
        self.assertGreater(ev.evaluate(worse).cost, base)


class _Quadratic:
    """Convex dummy evaluator with the Evaluator interface."""

    def __init__(self, target):
        self.target = target
        self.n_evals = 0

    def evaluate(self, dv):
        self.n_evals += 1
        cost = sum((dv[k] - t) ** 2 for k, t in self.target.items())
        return do.EvalResult(dv=dict(dv), metrics={}, violations={}, cost=cost)


class PatternSearchTests(unittest.TestCase):
    def test_converges_on_convex_quadratic(self):
        target = {name: 0.5 * (do.DESIGN_VARS[name][1] + do.DESIGN_VARS[name][2])
                  for name in do.VAR_ORDER}
        ev = _Quadratic(target)
        cfg = do.StudyConfig(quick=True)
        dv, best, history, rounds = do.pattern_search(ev, cfg)
        j0 = sum((do.BASELINE_DV[k] - t) ** 2 for k, t in target.items())
        self.assertLess(best.cost, j0)
        for name in do.VAR_ORDER:
            span = do.DESIGN_VARS[name][2] - do.DESIGN_VARS[name][1]
            self.assertLessEqual(abs(dv[name] - target[name]), 0.25 * span)
            lo, hi = do.DESIGN_VARS[name][1], do.DESIGN_VARS[name][2]
            self.assertGreaterEqual(dv[name], lo)
            self.assertLessEqual(dv[name], hi)
        self.assertGreater(len(history), 1)
        self.assertGreater(rounds, 0)

    def test_deterministic(self):
        target = {name: do.BASELINE_DV[name] + 0.05 for name in do.VAR_ORDER}
        cfg = do.StudyConfig(quick=True)
        dv1, best1, _, _ = do.pattern_search(_Quadratic(target), cfg)
        dv2, best2, _, _ = do.pattern_search(_Quadratic(target), cfg)
        self.assertEqual(dv1, dv2)
        self.assertEqual(best1.cost, best2.cost)


class RunStudyTests(unittest.TestCase):
    def test_end_to_end_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = do.StudyConfig(
                out=Path(tmp) / "study",
                takeoff_duration=MICRO.takeoff_duration,
                takeoff_dt=MICRO.takeoff_dt,
                landing_duration=MICRO.landing_duration,
                landing_dt=MICRO.landing_dt,
                final_landing_duration=MICRO.final_landing_duration,
                final_landing_dt=MICRO.final_landing_dt,
                initial_step_frac=MICRO.initial_step_frac,
                min_step_frac=MICRO.min_step_frac,
                max_rounds=MICRO.max_rounds,
                plots=False)
            summary = do.run_study(cfg)
            out = cfg.out
            for name in ("REPORT.md", "summary.json", "best_design.json",
                         "convergence.csv", "sensitivity.csv",
                         "design_optimize.py"):
                self.assertTrue((out / name).exists(), name)
            loaded = json.loads((out / "summary.json").read_text("utf-8"))
            for key in ("study", "train", "eval", "finding", "artifacts",
                        "caveats"):
                self.assertIn(key, loaded)
            self.assertEqual(summary["eval"]["J_best"],
                             loaded["eval"]["J_best"])
            self.assertLessEqual(loaded["eval"]["J_best"],
                                 loaded["eval"]["J_baseline"])
            design = json.loads((out / "best_design.json").read_text("utf-8"))
            for name, spec in do.DESIGN_VARS.items():
                v = design["design_variables"][name]
                self.assertGreaterEqual(v, spec[1])
                self.assertLessEqual(v, spec[2])
            report = (out / "REPORT.md").read_text("utf-8")
            self.assertIn("設計変数", report)
            self.assertIn("考察", report)


if __name__ == "__main__":
    unittest.main()
