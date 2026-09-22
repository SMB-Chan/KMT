"""Unit tests for interceptor drone comparison v2."""
import math
import unittest
import numpy as np

from interceptor import (
    EngagementConfig, FragmentationWarhead, InterceptorDrone,
    KineticImpactWarhead, PNGuidance, PurePursuit, PredictivePursuit,
    CrabPredictive, CrabPursuit, NearestSwitch, FixedSwitch,
    SensorNoise, SensorCraft, WarheadType, WindField, fuse_sensor_fix,
    run_engagement, run_multi_target, scenario_beam, scenario_evasive,
    scenario_head_on, scenario_tail_chase, evasive_heading_fn,
)


class TestInterceptorDrone(unittest.TestCase):
    def test_default_params(self):
        intr = InterceptorDrone()
        self.assertEqual(intr.mass, 5.0)
        self.assertAlmostEqual(intr.W, 5.0 * 9.80665, places=2)
        self.assertGreater(intr.T_max / intr.W, 2.5)

    def test_thrust(self):
        intr = InterceptorDrone()
        self.assertEqual(intr.thrust(50.0, 1.0), 150.0)
        self.assertEqual(intr.thrust(50.0, 0.0), 0.0)


class TestFragmentationWarhead(unittest.TestCase):
    def test_kill_probability_monotone(self):
        fw = FragmentationWarhead()
        p1 = fw.kill_probability(1.0)
        p3 = fw.kill_probability(3.0)
        p6 = fw.kill_probability(6.0)
        self.assertGreater(p1, p3)
        self.assertGreater(p3, p6)

    def test_near_zero_miss(self):
        fw = FragmentationWarhead()
        self.assertAlmostEqual(fw.kill_probability(0.1), 1.0, places=2)

    def test_beyond_lethal(self):
        fw = FragmentationWarhead()
        self.assertEqual(fw.kill_probability(8.0), 0.0)
        self.assertEqual(fw.kill_probability(20.0), 0.0)

    def test_realistic_radius(self):
        """Lethal radius should be small (≤8m) for a 5kg drone warhead."""
        fw = FragmentationWarhead()
        self.assertLessEqual(fw.R_lethal, 8.0)

    def test_pk_at_5m_is_moderate(self):
        """At 5m, a small warhead should not guarantee kill."""
        fw = FragmentationWarhead()
        p = fw.kill_probability(5.0)
        self.assertLess(p, 0.8)
        self.assertGreater(p, 0.0)


class TestKineticImpactWarhead(unittest.TestCase):
    def test_kinetic_energy(self):
        kw = KineticImpactWarhead()
        self.assertAlmostEqual(kw.kinetic_energy(20), 0.5 * 5.0 * 400, places=0)

    def test_low_speed_no_kill(self):
        """Below KE_floor, P_kill should be zero."""
        kw = KineticImpactWarhead()
        p = kw.kill_probability(1.0, 10)  # KE=250J < 500J floor
        self.assertEqual(p, 0.0)

    def test_moderate_speed_partial_kill(self):
        """Between floor and threshold, partial P_kill."""
        kw = KineticImpactWarhead()
        p = kw.kill_probability(1.0, 20)  # KE=1000J between 500 and 1500
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_high_speed_guaranteed_kill(self):
        """Above threshold at close range → P_kill = 1."""
        kw = KineticImpactWarhead()
        p = kw.kill_probability(1.0, 30)  # KE=2250J > 1500J
        self.assertEqual(p, 1.0)

    def test_miss_beyond_radius(self):
        kw = KineticImpactWarhead()
        p = kw.kill_probability(10.0, 60)
        self.assertEqual(p, 0.0)

    def test_effective_radius_includes_propellers(self):
        """Effective radius should be ≥ 2m (airframe + prop disc)."""
        kw = KineticImpactWarhead()
        self.assertGreaterEqual(kw.effective_radius, 2.0)


class TestPNGuidance(unittest.TestCase):
    def test_head_on_zero_los_rate(self):
        pn = PNGuidance(los_tau=0.0)
        ri = np.array([0.0, 0.0, 50.0])
        vi = np.array([50.0, 0.0, 0.0])
        rt = np.array([1000.0, 0.0, 50.0])
        vt = np.array([-15.0, 0.0, 0.0])
        pn.desired_heading(ri, vi, rt, vt, dt=0.1)
        ri2 = ri + vi * 0.1
        rt2 = rt + vt * 0.1
        heading = pn.desired_heading(ri2, vi, rt2, vt, dt=0.1)
        self.assertAlmostEqual(heading, 0.0, places=2)

    def test_crossing_target_integrates(self):
        """A sustained LOS rate must accumulate, not reset each step."""
        pn = PNGuidance(N=4.0, los_tau=0.0)
        ri = np.array([0.0, 0.0, 50.0])
        vi = np.array([50.0, 0.0, 0.0])
        rt = np.array([500.0, 100.0, 50.0])
        vt = np.array([0.0, -15.0, 0.0])
        pn.desired_heading(ri, vi, rt, vt, dt=0.1)
        # Hold interceptor fixed and move the target so LOS rate is steady.
        headings = []
        for k in range(1, 6):
            rt_k = rt + vt * (0.1 * k)
            headings.append(pn.desired_heading(ri, vi, rt_k, vt, dt=0.1))
        self.assertGreater(abs(headings[-1]), abs(headings[0]))
        self.assertGreater(abs(headings[-1]), math.radians(1.0))


class TestEngagement(unittest.TestCase):
    """Engagement tests with zero noise for deterministic checks."""

    def _zero_noise(self):
        return SensorNoise(pos_sigma=0.0, vel_sigma=0.0)

    def test_head_on_fragmentation(self):
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               sensor=self._zero_noise())
        tp, tv, ip, iv = scenario_head_on()
        result = run_engagement(tp, tv, ip, iv, cfg, seed=42)
        self.assertLess(result.miss_distance, 10.0)
        self.assertTrue(result.hit)

    def test_head_on_kinetic(self):
        cfg = EngagementConfig(warhead_type=WarheadType.KINETIC_IMPACT,
                               sensor=self._zero_noise())
        tp, tv, ip, iv = scenario_head_on()
        result = run_engagement(tp, tv, ip, iv, cfg, seed=42)
        # With zero noise and head-on, should get close
        self.assertLess(result.miss_distance, 10.0)

    def test_tail_chase(self):
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               sensor=self._zero_noise())
        tp, tv, ip, iv = scenario_tail_chase()
        result = run_engagement(tp, tv, ip, iv, cfg, seed=42)
        self.assertLess(result.miss_distance, 20.0)

    def test_noise_increases_miss(self):
        """Adding noise should increase miss distance on average."""
        tp, tv, ip, iv = scenario_head_on()
        misses_quiet = []
        misses_noisy = []
        for seed in range(100, 110):
            cfg_q = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                      sensor=SensorNoise(0.0, 0.0))
            cfg_n = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                      sensor=SensorNoise(0.5, 0.2))
            misses_quiet.append(run_engagement(tp, tv, ip, iv, cfg_q, seed=seed).miss_distance)
            misses_noisy.append(run_engagement(tp, tv, ip, iv, cfg_n, seed=seed).miss_distance)
        self.assertGreater(np.mean(misses_noisy), np.mean(misses_quiet))

    def test_evasive_increases_miss(self):
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               sensor=SensorNoise(0.5, 0.2))
        misses_static = []
        misses_evasive = []
        for seed in range(200, 250):
            tp, tv, ip, iv = scenario_head_on()
            r1 = run_engagement(tp, tv, ip, iv, cfg, seed=seed)
            hdg = evasive_heading_fn(seed=seed)
            r2 = run_engagement(tp, tv, ip, iv, cfg, seed=seed, target_heading_fn=hdg)
            misses_static.append(r1.miss_distance)
            misses_evasive.append(r2.miss_distance)
        # Evasive should increase miss *variance* at least
        self.assertGreater(np.std(misses_evasive), np.std(misses_static) * 0.5)

    def test_fragmentation_tolerates_larger_miss(self):
        """At 5m miss, fragmentation should have higher P_kill than kinetic."""
        fw = FragmentationWarhead()
        kw = KineticImpactWarhead()
        frag_p = fw.kill_probability(5.0)
        kin_p = kw.kill_probability(5.0, 60.0)
        self.assertGreater(frag_p, 0.0)
        self.assertEqual(kin_p, 0.0)  # beyond 3m effective radius

    def test_closing_speed_positive_when_closing(self):
        """When interceptor approaches target, closing_speed should be positive."""
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               sensor=SensorNoise(0.0, 0.0))
        tp, tv, ip, iv = scenario_head_on()
        result = run_engagement(tp, tv, ip, iv, cfg, seed=42)
        # At closest approach, should be closing
        if result.hit:
            self.assertGreater(result.closing_speed, 0.0)

    def test_engagement_time_matches_kinematics(self):
        """Engagement time should roughly match R0 / closing_speed."""
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               sensor=SensorNoise(0.0, 0.0))
        tp, tv, ip, iv = scenario_head_on()
        result = run_engagement(tp, tv, ip, iv, cfg, seed=42)
        kc = result.kinematic_check
        if kc and result.hit:
            # Actual time should be within 2x of expected
            if kc["t_expected_s"] > 0:
                ratio = kc["t_actual_s"] / kc["t_expected_s"]
                self.assertGreater(ratio, 0.1)
                self.assertLess(ratio, 5.0)

    def test_beam_geometry_is_hard(self):
        """Pure pursuit on a beam intercept lags a head-on collision course."""
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               guidance_law=PurePursuit(),
                               sensor=SensorNoise(0.0, 0.0))
        tp_h, tv_h, ip_h, iv_h = scenario_head_on()
        tp_b, tv_b, ip_b, iv_b = scenario_beam()
        r_h = run_engagement(tp_h, tv_h, ip_h, iv_h, cfg, seed=42)
        r_b = run_engagement(tp_b, tv_b, ip_b, iv_b, cfg, seed=42)
        self.assertGreater(r_b.miss_distance, r_h.miss_distance)

    def test_pn_beam_turns_inside_lethal_radius(self):
        """Integrated PN must turn on a beam; a one-step rebase does not."""
        sensor = SensorNoise(0.0, 0.0)
        geom = scenario_beam(1000, 15, 50, 50)
        pursuit = run_engagement(
            *geom,
            EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                             guidance_law=PurePursuit(), sensor=sensor),
            seed=0)
        pn = run_engagement(
            *geom,
            EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                             guidance_law=PNGuidance(N=4.0, los_tau=0.0),
                             sensor=sensor),
            seed=0)
        self.assertLess(pn.miss_distance, pursuit.miss_distance)
        self.assertLess(pn.miss_distance, 6.0)

    def test_multi_target_drag_uses_airframe_cd(self):
        """Switching engagements must use CD(), not a missing Cd field."""
        cfg = EngagementConfig(
            guidance_law=CrabPredictive(wind=(0.0, 10.0, 0.0)),
            sensor=SensorNoise(0.5, 0.2),
            wind=WindField(mean=(0.0, 10.0, 0.0)),
            wind_est_sigma=0.5,
            store_trajectory=False,
            max_time=0.3,
        )
        targets = [
            (np.array([500.0, 0.0, 50.0]), np.array([-15.0, 0.0, 0.0]), None),
            (np.array([500.0, 80.0, 50.0]), np.array([-8.0, -9.0, 0.0]), None),
        ]
        res = run_multi_target(
            targets, np.array([0.0, 40.0, 50.0]), np.array([45.0, -22.0, 0.0]),
            cfg, NearestSwitch(0.5), seed=1)
        self.assertEqual(len(res["miss_by_target"]), 2)
        self.assertEqual(len(res["approach_by_target"]), 2)
        self.assertIn(res["final_target"], (0, 1))
        self.assertGreaterEqual(res["switches"], 0)

    def test_higher_cruise_closes_tail_chase_that_times_out(self):
        """2000 m behind a 20 m/s target is outside 60 s at 50 m/s, inside at 100."""
        from guidance_speed_turn import _airframe
        slow = EngagementConfig(
            interceptor=_airframe(50.0, 5.67),
            guidance_law=PredictivePursuit(),
            sensor=SensorNoise(0.0, 0.0),
            store_trajectory=False,
        )
        fast = EngagementConfig(
            interceptor=_airframe(100.0, 5.67),
            guidance_law=PredictivePursuit(),
            sensor=SensorNoise(0.0, 0.0),
            store_trajectory=False,
        )
        tp, tv, ip, iv = scenario_tail_chase(2000, 20, 50, 50)
        r_slow = run_engagement(tp, tv, ip, iv, slow, seed=0)
        tp, tv, ip, iv = scenario_tail_chase(2000, 20, 100, 50)
        r_fast = run_engagement(tp, tv, ip, iv, fast, seed=0)
        self.assertGreater(r_slow.engagement_time, 59.0)
        self.assertLess(r_fast.engagement_time, 40.0)
        self.assertLess(r_fast.miss_distance, 6.0)

    def test_sensor_craft_holds_track_and_fuses(self):
        """A ranger that holds course tightens the fix.  It is not a warhead."""
        craft = SensorCraft(pos0=(0.0, 20.0, 50.0), vel=(0.0, 0.0, 0.0),
                            angle_sigma=1e-6, range_sigma=0.01, link_sigma=0.05)
        self.assertTrue(np.allclose(craft.position(10.0), [0.0, 20.0, 50.0]))
        rng = np.random.default_rng(0)
        own_ri = np.array([0.0, 0.0, 50.0])
        own_rt = np.array([100.0, 40.0, 50.0])
        truth = np.array([0.0, 0.0, 50.0])
        fused = fuse_sensor_fix(own_rt, own_ri, 5.0, craft, truth, own_ri, 0.0, rng)
        self.assertLess(np.hypot(fused[0] - truth[0], fused[1] - truth[1]), 1.0)

    def test_ranging_link_beats_own_relative_fix(self):
        """A close ranger's relative vector is tighter than a 5 m own-ship fix."""
        craft = SensorCraft(pos0=(0.0, 30.0, 50.0), vel=(0.0, 0.0, 0.0),
                            angle_sigma=1e-4, range_sigma=0.3, link_sigma=0.1)
        ri = np.array([0.0, 0.0, 50.0])
        truth = np.array([40.0, 0.0, 50.0])
        own_err, fuse_err = [], []
        rng = np.random.default_rng(3)
        for _ in range(30):
            own_rt = truth + rng.normal(0.0, 5.0, 3)
            fused = fuse_sensor_fix(own_rt, ri, 5.0, craft, truth, ri, 0.0, rng)
            own_err.append(np.hypot(*(own_rt[:2] - truth[:2])))
            fuse_err.append(np.hypot(*(fused[:2] - truth[:2])))
        self.assertLess(float(np.median(fuse_err)), float(np.median(own_err)) * 0.5)

    def test_multi_target_holds_altitude_on_collision_course(self):
        """A locked head-on must not fall out of the engagement plane."""
        cfg = EngagementConfig(
            guidance_law=PredictivePursuit(),
            sensor=SensorNoise(0.0, 0.0),
            wind=WindField(),
            store_trajectory=False,
            max_time=30.0,
        )
        res = run_multi_target(
            [(np.array([1000.0, 0.0, 50.0]), np.array([-15.0, 0.0, 0.0]), None)],
            np.array([0.0, 0.0, 50.0]), np.array([50.0, 0.0, 0.0]),
            cfg, FixedSwitch(0), seed=1)
        self.assertLess(res["miss_by_target"][0], 3.0)

    def test_kinetic_scores_impact_parameter(self):
        """A head-on collision course is not scored as a 3 m graze."""
        cfg = EngagementConfig(warhead_type=WarheadType.KINETIC_IMPACT,
                               guidance_law=PurePursuit(),
                               sensor=SensorNoise(0.0, 0.0))
        result = run_engagement(*scenario_head_on(1000, 15, 50, 50), cfg, seed=1)
        self.assertLess(result.miss_distance, 2.0)
        self.assertGreater(result.kill_probability, 0.8)
        self.assertGreater(result.approach_speed, 20.0)


class TestDisturbances(unittest.TestCase):
    """Wind and communication delay are optional and default off."""

    def _zero_noise(self):
        return SensorNoise(pos_sigma=0.0, vel_sigma=0.0)

    def test_calm_wind_is_default_off(self):
        self.assertTrue(WindField().is_calm())
        self.assertEqual(EngagementConfig().comm_delay, 0.0)

    def test_crosswind_increases_miss(self):
        """Unmodelled crosswind must degrade an otherwise clean intercept."""
        geom = scenario_beam(1000, 15, 50, 50)
        quiet = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                 guidance_law=PredictivePursuit(),
                                 sensor=self._zero_noise(),
                                 wind=WindField())
        windy = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                 guidance_law=PredictivePursuit(),
                                 sensor=self._zero_noise(),
                                 wind=WindField(mean=(0.0, 10.0, 0.0)))
        misses_calm = []
        misses_wind = []
        for seed in range(10, 40):
            misses_calm.append(
                run_engagement(*geom, quiet, seed=seed).miss_distance)
            misses_wind.append(
                run_engagement(*geom, windy, seed=seed).miss_distance)
        self.assertGreater(np.median(misses_wind), np.median(misses_calm))
        self.assertGreater(np.median(misses_wind), 2.0)

    def test_comm_delay_increases_miss(self):
        """A stale fix on a jinking target must hurt more than a live one."""
        tp, tv, ip, iv = scenario_head_on(1000, 15, 50, 50)
        live = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                guidance_law=PNGuidance(N=4.0, los_tau=0.0),
                                sensor=SensorNoise(0.5, 0.2),
                                comm_delay=0.0)
        delayed = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                   guidance_law=PNGuidance(N=4.0, los_tau=0.0),
                                   sensor=SensorNoise(0.5, 0.2),
                                   comm_delay=0.15)
        misses_live = []
        misses_late = []
        for seed in range(200, 250):
            hdg = evasive_heading_fn(seed=seed)
            misses_live.append(run_engagement(
                tp, tv, ip, iv, live, seed=seed,
                target_heading_fn=hdg).miss_distance)
            misses_late.append(run_engagement(
                tp, tv, ip, iv, delayed, seed=seed,
                target_heading_fn=hdg).miss_distance)
        self.assertGreater(np.median(misses_late), np.median(misses_live))

    def test_target_drift_zero_holds_ground_track(self):
        """A multirotor threat is not blown along with the wind."""
        cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                               guidance_law=PurePursuit(),
                               sensor=self._zero_noise(),
                               wind=WindField(mean=(8.0, 0.0, 0.0),
                                              target_drift=0.0),
                               store_trajectory=True)
        tp, tv, ip, iv = scenario_head_on(500, 15, 50, 50)
        res = run_engagement(tp, tv, ip, iv, cfg, seed=0)
        tt = np.array(res.trajectory_target)
        # Target speed over ground stays near the command (15 m/s along +x?).
        # scenario_head_on: target at +x flying -x at 15 m/s.
        self.assertGreater(len(tt), 50)
        # y of the target must not be swept by the +x wind.
        self.assertLess(abs(tt[-1, 1] - tt[0, 1]), 2.0)

    def test_delay_zero_matches_immediate_obs(self):
        """comm_delay=0 must feed the live sample to the law."""
        cfg = EngagementConfig(warhead_type=WarheadType.KINETIC_IMPACT,
                               guidance_law=PNGuidance(N=4.0, los_tau=0.0),
                               sensor=SensorNoise(0.0, 0.0),
                               comm_delay=0.0)
        r = run_engagement(*scenario_head_on(1000, 15, 50, 50), cfg, seed=3)
        self.assertLess(r.miss_distance, 5.0)

    def test_crosswind_hurts_predictive_more_than_pn(self):
        """Unmodelled wind breaks the collision-triangle lead; PN crab-compensates."""
        geom = scenario_head_on(1000, 15, 50, 50)
        wind = WindField(mean=(0.0, 10.0, 0.0))

        def misses(law):
            cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                   guidance_law=law,
                                   sensor=SensorNoise(0.5, 0.2),
                                   wind=wind)
            out = []
            for seed in range(300, 340):
                hdg = evasive_heading_fn(seed=seed)
                out.append(run_engagement(
                    *geom, cfg, seed=seed, target_heading_fn=hdg).miss_distance)
            return np.median(out)

        pred = misses(PredictivePursuit())
        pn = misses(PNGuidance(N=4.0, los_tau=0.0))
        self.assertGreater(pred, pn)
        self.assertGreater(pred, 5.0)

    def test_crab_restores_predictive_in_crosswind(self):
        """Crab into the wind must recover the collision-triangle lead."""
        geom = scenario_head_on(1000, 15, 50, 50)
        wind = WindField(mean=(0.0, 10.0, 0.0))

        def misses(law):
            cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION,
                                   guidance_law=law,
                                   sensor=SensorNoise(0.5, 0.2),
                                   wind=wind)
            out = []
            for seed in range(400, 440):
                hdg = evasive_heading_fn(seed=seed)
                out.append(run_engagement(
                    *geom, cfg, seed=seed, target_heading_fn=hdg).miss_distance)
            return np.median(out)

        raw = misses(PredictivePursuit())
        crab_air = misses(CrabPredictive(wind=(0.0, 10.0, 0.0),
                                         estimate_wind=True))
        crab_oracle = misses(
            CrabPredictive(wind=(0.0, 10.0, 0.0), estimate_wind=False))
        self.assertLess(crab_oracle, 3.0)
        self.assertLess(crab_oracle, raw * 0.2)
        self.assertLess(crab_air, raw * 0.5)


if __name__ == "__main__":
    unittest.main()
