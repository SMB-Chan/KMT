import unittest
import numpy as np
from aircraft import Aircraft
from env import EnvConfig, FlyingBoatEnv
from train import teacher_action


class RobustTrainingTests(unittest.TestCase):
    def test_conditions_are_reproducible_and_config_is_not_mutated(self):
        cfg = EnvConfig(spatial=True, directional=True, randomize_conditions=True)
        env = FlyingBoatEnv(Aircraft(), cfg)
        first = env.reset(seed=9)
        conditions = env.episode_conditions.copy()
        np.testing.assert_array_equal(first, env.reset(seed=9))
        self.assertEqual(conditions, env.episode_conditions)
        env.reset(seed=10)
        self.assertNotEqual(conditions, env.episode_conditions)
        self.assertEqual(cfg.Hs, 1.5)
        self.assertEqual(cfg.atmosphere.wind, (0, 0, 0))

    def test_four_axis_teacher_corrects_lateral_offset(self):
        for scenario in ('takeoff', 'landing'):
            cfg = EnvConfig(spatial=True, scenario=scenario)
            env = FlyingBoatEnv(Aircraft(), cfg)
            state = env.reset(seed=0)
            offset = 8 + len(cfg.preview_dx_m)
            state[offset] = 0.5
            action = teacher_action(state, cfg, env.ac)
            self.assertEqual(action.shape, (4,))
            self.assertTrue(np.isfinite(action).all())
            self.assertLess(action[2], 0)
            self.assertLess(action[3], 0)
            env.step(action)

class CloningConvergenceTests(unittest.TestCase):
    def test_action_space_loss_decreases_without_changing_critic_head(self):
        from policy import ActorCritic
        model = ActorCritic(3, 4, np.array([0, -1, -1, -1]), np.ones(4), hidden=8, seed=2)
        states = np.array([[0, 1, 0], [1, 0, 1], [-1, 1, 1]], dtype=np.float32)
        actions = np.array([[1, .2, -.2, .1], [0, -.4, .3, -.2], [.5, .1, 0, 0]], dtype=np.float32)
        critic = model.critic_head_W.copy()
        before = model.imitate(states, actions, lr=0, action_space=True)['bc_loss']
        for _ in range(100):
            model.imitate(states, actions, lr=.05, action_space=True)
        after = model.imitate(states, actions, lr=0, action_space=True)['bc_loss']
        self.assertLess(after, before * .9)
        np.testing.assert_array_equal(critic, model.critic_head_W)

    def test_action_space_gradient_matches_finite_difference(self):
        from policy import ActorCritic
        model = ActorCritic(2, 2, np.array([0, -1]), np.ones(2), hidden=4, seed=3)
        states = np.array([[.2, -.4], [.7, .1]], dtype=np.float32)
        actions = np.array([[1, .3], [.2, -.1]], dtype=np.float32)
        original = float(model.actor_head_W[0, 1])
        eps = .001
        model.actor_head_W[0, 1] = original + eps
        plus = model.imitate(states, actions, lr=0, action_space=True)['bc_loss']
        model.actor_head_W[0, 1] = original - eps
        minus = model.imitate(states, actions, lr=0, action_space=True)['bc_loss']
        model.actor_head_W[0, 1] = original
        model.imitate(states, actions, lr=.01, action_space=True)
        analytic = (original - float(model.actor_head_W[0, 1])) / .01
        self.assertAlmostEqual(analytic, (plus - minus) / (2 * eps), delta=1e-4)
