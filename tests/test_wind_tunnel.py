"""Tests for wind_tunnel.py (atmosphere-driven virtual wind tunnel).

Guards:
  * the tunnel test section reproduces atmosphere.py ISA / shear / gusts,
  * polar reduction recovers the airframe's aerodynamic constants,
  * gust load series are deterministic (same seed -> bitwise identical)
    and degenerate to n == 1 without turbulence,
  * the tunnel objective pins J(baseline) = 1 and the compass search is
    deterministic and never worse than baseline,
  * the end-to-end study writes all artifacts (quick budget, tmp dir).
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import wind_tunnel as wt
from aircraft import Aircraft, RHO
from atmosphere import Atmosphere
from design_optimize import BASELINE_DV, DESIGN_VARS

MICRO = wt.TunnelConfig(quick=True, plots=False)


def micro_cfg(out: Path) -> wt.TunnelConfig:
    return wt.TunnelConfig(out=out, quick=True, plots=False)


class AtmosphereInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.atm = Atmosphere(seed=0)

    def test_sea_level_density_pinned_to_rho(self):
        self.assertAlmostEqual(self.atm.density(0.0), RHO, places=12)

    def test_isa_profile_monotone(self):
        hs = list(MICRO.altitudes)
        rho = [self.atm.density(h) for h in hs]
        temp = [self.atm.temperature(h) for h in hs]
        self.assertTrue(all(a > b for a, b in zip(rho, rho[1:])))
        self.assertTrue(all(a > b for a, b in zip(temp, temp[1:])))

    def test_conditions_carry_reynolds_and_mach(self):
        c = wt.tunnel_conditions(self.atm, 0.0)
        self.assertAlmostEqual(c["sound_speed_m_s"],
                               math.sqrt(1.4 * 287.05287 * 288.15), places=6)
        self.assertAlmostEqual(wt.reynolds(1.225, 10.0, 1.5, 1.789e-5),
                               1.225 * 10.0 * 1.5 / 1.789e-5, places=3)
        self.assertAlmostEqual(wt.mach_number(c["sound_speed_m_s"],
                                              c["temperature_K"]), 1.0)

    def test_shear_factor_reference(self):
        self.assertAlmostEqual(self.atm.shear_factor(10.0), 1.0, places=12)
        self.assertLess(self.atm.shear_factor(1.0), 1.0)
        self.assertGreater(self.atm.shear_factor(100.0), 1.0)


class PolarReductionTests(unittest.TestCase):
    def setUp(self):
        self.ac = wt.apply_design(BASELINE_DV)
        self.rows = wt.polar_sweep(self.ac, wt.alpha_grid(MICRO))
        self.meas = wt.measure_polar(self.rows)

    def test_baseline_reproduces_aircraft(self):
        self.assertEqual(self.ac, Aircraft())

    def test_lift_slope_and_cl0_recovered(self):
        self.assertAlmostEqual(self.meas["CL_alpha_rad"],
                               self.ac.CL_alpha_3d, places=9)
        self.assertAlmostEqual(self.meas["CL0"], self.ac.aero.CL0, places=9)

    def test_clmax_and_stall_angle(self):
        self.assertAlmostEqual(self.meas["CL_max_meas"],
                               self.ac.aero.CL_max, places=9)
        theory = math.degrees((self.ac.aero.CL_max - self.ac.aero.CL0)
                              / self.ac.CL_alpha_3d)
        self.assertLessEqual(abs(self.meas["alpha_stall_deg"] - theory),
                             MICRO.alpha_step_deg)

    def test_drag_polar_coefficients_recovered(self):
        k_theory = 1.0 / (math.pi * self.ac.aero.e * self.ac.geom.AR)
        self.assertAlmostEqual(self.meas["CD0_meas"], self.ac.aero.CD0,
                               places=12)
        self.assertAlmostEqual(self.meas["K_meas"], k_theory, places=12)

    def test_ld_max_close_to_analytic(self):
        rel = abs(self.meas["LD_max"] - self.meas["LD_max_analytic"])
        self.assertLess(rel / self.meas["LD_max_analytic"], 0.01)

    def test_ground_effect_relief_monotone(self):
        ge = wt.ground_effect_sweep(self.ac, RHO, MICRO)
        red = [r["D_reduction_pct"] for r in ge]
        self.assertTrue(all(a > b for a, b in zip(red, red[1:])))
        self.assertGreater(red[0], 5.0)
        self.assertLess(red[-1], 0.1)
        for r in ge:
            self.assertAlmostEqual(r["ge_factor"],
                                   self.ac.induced_drag_factor(r["h_m"]),
                                   places=12)


class VelocitySweepTests(unittest.TestCase):
    def setUp(self):
        self.ac = wt.apply_design(BASELINE_DV)
        self.rows = wt.velocity_sweep(self.ac, RHO, MICRO)
        self.lm = wt.sweep_landmarks(self.rows)

    def test_trim_balance(self):
        r = self.rows[0]
        q = 0.5 * RHO * r["V_m_s"] ** 2
        self.assertAlmostEqual(r["CL_trim"], self.ac.W / (q * self.ac.geom.S),
                               places=9)
        self.assertTrue(all(r["feasible"] for r in self.rows))

    def test_landmarks_ordered(self):
        self.assertLess(self.lm["V_minpower_m_s"], self.lm["V_mindrag_m_s"])
        self.assertLess(self.lm["V_minpower_m_s"], self.lm["V_maxlevel_m_s"])
        self.assertLess(self.lm["V_maxlevel_m_s"], self.rows[-1]["V_m_s"])

    def test_thrust_margin_sign_change(self):
        margins = [r["thrust_margin_N"] for r in self.rows]
        self.assertGreater(margins[0], 0.0)
        self.assertLess(margins[-1], 0.0)


class GustTunnelTests(unittest.TestCase):
    def setUp(self):
        self.ac = wt.apply_design(BASELINE_DV)

    def test_no_turbulence_gives_unit_load(self):
        atm = wt.make_gust_atmosphere(MICRO, "dryden", 0.0)
        g = wt.gust_load_stats(self.ac, atm, altitude=30.0,
                               V=self.ac.V_cruise, duration=5.0, dt=0.05)
        self.assertAlmostEqual(g["n_std"], 0.0, places=12)
        self.assertAlmostEqual(g["n_mean"], 1.0, places=9)

    def test_same_seed_bitwise_reproducible(self):
        a = wt.gust_run(self.ac, MICRO, "dryden", 2.0)
        b = wt.gust_run(self.ac, MICRO, "dryden", 2.0)
        self.assertTrue(np.array_equal(a["n"], b["n"]))
        self.assertTrue(np.array_equal(a["alpha_eff"], b["alpha_eff"]))

    def test_different_seed_changes_series(self):
        a = wt.gust_run(self.ac, MICRO, "dryden", 2.0)
        c = wt.make_gust_atmosphere(MICRO, "dryden", 2.0, seed=7)
        g = wt.gust_load_stats(self.ac, c, altitude=MICRO.cruise_altitude,
                               V=self.ac.V_cruise, duration=MICRO.gust_duration,
                               dt=MICRO.gust_dt)
        self.assertFalse(np.array_equal(a["n"], g["n"]))

    def test_load_rms_grows_with_turbulence(self):
        std = [wt.gust_run(self.ac, MICRO, "dryden", rms)["n_std"]
               for rms in MICRO.gust_rms_levels]
        self.assertTrue(all(a < b for a, b in zip(std, std[1:])))

    def test_shear_shields_near_surface(self):
        hi = wt.gust_run(self.ac, MICRO, "dryden", 1.0)
        lo = wt.gust_run(self.ac, MICRO, "dryden", 1.0, altitude=1.0)
        self.assertLess(lo["n_std"], hi["n_std"])

    def test_gust_models_differ(self):
        s = wt.gust_run(self.ac, MICRO, "sum4", 2.0)
        d = wt.gust_run(self.ac, MICRO, "dryden", 2.0)
        self.assertFalse(np.array_equal(s["n"], d["n"]))


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MICRO
        self.atm = wt.make_gust_atmosphere(self.cfg, "dryden",
                                            self.cfg.opt_gust_rms)
        self.ev = wt.TunnelEvaluator(self.cfg, self.atm)

    def test_baseline_cost_is_one(self):
        r = self.ev.evaluate(BASELINE_DV)
        self.assertEqual(r.violations, {})
        self.assertAlmostEqual(r.cost, sum(wt.WEIGHTS.values()), places=12)

    def test_evaluation_deterministic(self):
        a = self.ev.evaluate(BASELINE_DV)
        b = self.ev.evaluate(BASELINE_DV)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.metrics, b.metrics)

    def test_degraded_design_costs_more(self):
        worse = {**BASELINE_DV, "CD0": DESIGN_VARS["CD0"][2],
                 "S_scale": DESIGN_VARS["S_scale"][1],
                 "b_scale": DESIGN_VARS["b_scale"][1]}
        self.assertGreater(self.ev.evaluate(worse).cost,
                           self.ev.evaluate(BASELINE_DV).cost)

    def test_search_deterministic_and_improving(self):
        atm2 = wt.make_gust_atmosphere(self.cfg, "dryden",
                                       self.cfg.opt_gust_rms)
        ev2 = wt.TunnelEvaluator(self.cfg, atm2)
        dv1, best1, hist1, rounds1 = wt.compass_search(self.ev, self.cfg)
        dv2, best2, hist2, rounds2 = wt.compass_search(ev2, self.cfg)
        self.assertEqual(dv1, dv2)
        self.assertEqual(best1.cost, best2.cost)
        self.assertEqual([h["cost"] for h in hist1],
                         [h["cost"] for h in hist2])
        self.assertEqual(rounds1, rounds2)
        self.assertLessEqual(best1.cost, sum(wt.WEIGHTS.values()))
        for name in wt.TUNNEL_VARS:
            lo, hi = DESIGN_VARS[name][1], DESIGN_VARS[name][2]
            self.assertTrue(lo - 1e-12 <= dv1[name] <= hi + 1e-12)

    def test_sensitivity_rows_cover_tunnel_vars(self):
        _, best, _, _ = wt.compass_search(self.ev, self.cfg)
        rows = wt.tunnel_sensitivity(self.ev, best.dv)
        self.assertEqual([r["var"] for r in rows], list(wt.TUNNEL_VARS))
        for r in rows:
            self.assertGreaterEqual(r["J_range"], 0.0)
            self.assertAlmostEqual(r["J_center"], best.cost, places=12)


class EndToEndTests(unittest.TestCase):
    def test_study_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = micro_cfg(Path(tmp) / "wt_test")
            summary = wt.run_study(cfg)
            names = {p.name for p in cfg.out.iterdir()}
            for n in ("REPORT.md", "summary.json", "best_design.json",
                      "altitude_conditions.csv", "shear_profile.csv",
                      "polars.csv", "velocity_sweep.csv",
                      "ground_effect.csv", "gust_runs.csv",
                      "convergence.csv", "sensitivity.csv",
                      "wind_tunnel.py"):
                self.assertIn(n, names)
            for key in ("study", "train", "eval", "finding", "artifacts",
                        "caveats", "git_head", "runtime_s"):
                self.assertIn(key, summary)
            self.assertLessEqual(summary["eval"]["J_best"],
                                 summary["eval"]["J_baseline"] + 1e-9)
            self.assertEqual(summary["eval"]["metrics_baseline"]["S"], 22.5)

    def test_metrics_json_keys(self):
        atm = wt.make_gust_atmosphere(MICRO, "dryden", MICRO.opt_gust_rms)
        m = wt.tunnel_session_metrics(wt.apply_design(BASELINE_DV), atm,
                                      MICRO)
        self.assertEqual(set(wt.metrics_json(m)), set(wt.METRIC_KEYS))


if __name__ == "__main__":
    unittest.main()
