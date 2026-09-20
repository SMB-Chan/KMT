import math
import tempfile
import unittest
from pathlib import Path

from operator_training.runtime import (
    FixedStepScheduler,
    SimOverrun,
    Watchdog,
)


class WatchdogTests(unittest.TestCase):
    def test_no_lapse_when_fresh(self):
        wd = Watchdog(input_max_age_s=0.25, heartbeat_max_age_s=0.5)
        wd.note_input(t=100.0)
        wd.note_heartbeat(t=100.0)
        self.assertIsNone(wd.check(now=100.1))

    def test_input_stale_triggers(self):
        wd = Watchdog(input_max_age_s=0.25, heartbeat_max_age_s=0.5)
        wd.note_input(t=100.0)
        self.assertIsNotNone(wd.check(now=100.5))

    def test_heartbeat_stale_triggers(self):
        wd = Watchdog(input_max_age_s=0.25, heartbeat_max_age_s=0.5)
        wd.note_input(t=100.0)
        wd.note_heartbeat(t=100.0)
        self.assertIsNotNone(wd.check(now=101.0))

    def test_lapse_callback_invoked(self):
        called = []
        wd = Watchdog(input_max_age_s=0.25, on_lapse=called.append)
        wd.note_input(t=100.0)
        wd.check(now=100.5)
        wd.trigger("input_stale>=0.25s")
        self.assertEqual(called, ["input_stale>=0.25s"])


class FixedStepSchedulerTests(unittest.TestCase):
    def test_no_tick_when_no_elapsed_time(self):
        sched = FixedStepScheduler(dt=0.05)
        ticks = []
        n = sched.pump(now=100.0, feed=lambda dt, i: ticks.append(i))
        self.assertEqual(n, 0)
        self.assertEqual(ticks, [])

    def test_one_tick_per_50ms(self):
        sched = FixedStepScheduler(dt=0.05)
        ticks = []
        sched.pump(now=100.0, feed=lambda dt, i: ticks.append(i))
        n = sched.pump(now=100.06, feed=lambda dt, i: ticks.append(i))
        self.assertEqual(n, 1)
        self.assertEqual(ticks, [1])

    def test_catchup_caps_at_max(self):
        sched = FixedStepScheduler(dt=0.05, max_catchup_ticks=2)
        ticks = []
        sched.pump(now=100.0, feed=lambda dt, i: ticks.append(i))
        n = sched.pump(now=100.5, feed=lambda dt, i: ticks.append(i))
        # 10 ticks due, but max_catchup=2 -> only 2 ticks run
        self.assertEqual(n, 2)
        self.assertEqual(ticks, [1, 2])

    def test_overrun_callback_fires_on_huge_gap(self):
        overruns = []
        sched = FixedStepScheduler(
            dt=0.05, max_catchup_ticks=2,
            on_overrun=lambda dt: overruns.append(dt),
        )
        sched.pump(now=100.0, feed=lambda dt, i: None)
        sched.pump(now=101.0, feed=lambda dt, i: None)
        self.assertEqual(len(overruns), 1)

    def test_step_once_invokes_feed(self):
        sched = FixedStepScheduler(dt=0.05)
        ticks = []
        sched.step_once(feed=lambda dt, i: ticks.append(i))
        sched.step_once(feed=lambda dt, i: ticks.append(i))
        self.assertEqual(ticks, [1, 2])


class DeterminismTests(unittest.TestCase):
    """Re-running a session with the same seed must yield identical state."""

    def test_two_sessions_same_seed_same_altitude(self):
        from operator_training.session import (
            Session, SessionConfig, Curriculum,
        )
        from operator_training.envelope import ControlEnvelope
        from operator_training.recording import NullSink
        from operator_training.curriculum import CurriculumPhase

        def make():
            cfg = SessionConfig(
                curriculum=Curriculum(
                    curriculum_id="det",
                    curriculum_version="1",
                    scenario="hybrid",
                ),
                envelope=ControlEnvelope.beginner(),
                seed=42,
            )
            return Session.create(config=cfg, sink=NullSink(), seed=42)

        s1, s2 = make(), make()
        s1.ready(); s1.start_takeoff()
        s2.ready(); s2.start_takeoff()
        for i in range(50):
            s1.tick(dt=0.05, human_input=None, input_seq=i)
            s2.tick(dt=0.05, human_input=None, input_seq=i)
        self.assertEqual(s1.adapter.vehicle.z, s2.adapter.vehicle.z)
        self.assertEqual(s1.adapter.vehicle.Vx, s2.adapter.vehicle.Vx)


class DamagedVehicleTests(unittest.TestCase):
    def test_damage_finish(self):
        from operator_training.session import (
            Session, SessionConfig, Curriculum, Lifecycle,
        )
        from operator_training.envelope import ControlEnvelope
        from operator_training.recording import NullSink

        cfg = SessionConfig(
            curriculum=Curriculum(
                curriculum_id="dmg",
                curriculum_version="1",
                scenario="hybrid",
            ),
            envelope=ControlEnvelope.beginner(),
            seed=42,
        )
        sess = Session.create(config=cfg, sink=NullSink(), seed=42)
        sess.ready(); sess.start_takeoff()
        # Inject damage state directly
        sess.adapter.vehicle.damage.failed = True
        sess.adapter.vehicle.damage.failure_reason = "test_failure"
        sess.tick(dt=0.05, human_input=None)
        self.assertEqual(sess.state.lifecycle, Lifecycle.FINISHED)
        self.assertIn("damage", sess.state.end_reason)


if __name__ == "__main__":
    unittest.main()
