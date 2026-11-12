import numpy as np
import pytest

from skimmer.baselines import BASELINES
from skimmer.env import Config, OBS_DIM, REGIMES, SlickEnv
from skimmer.runner import run_episodes


def test_obs_shape_and_finite():
    env = SlickEnv(8, Config(), seed=1)
    obs = env.reset()
    assert obs.shape == (8, OBS_DIM) and np.isfinite(obs).all()
    for _ in range(5):
        obs, r, done, info = env.step(np.zeros(8, dtype=int))
        assert np.isfinite(obs).all() and np.isfinite(r).all()


def test_same_seed_is_reproducible():
    a, b = SlickEnv(4, seed=7), SlickEnv(4, seed=7)
    for _ in range(20):
        oa = a.step(np.full(4, 2))[0]
        ob = b.step(np.full(4, 2))[0]
    assert np.allclose(oa, ob)


def test_oil_only_decreases_and_ship_stays_in_world():
    env = SlickEnv(16, seed=3)
    prev = env.active.sum(axis=1)
    rng = np.random.default_rng(0)
    for _ in range(60):
        env.step(rng.integers(0, 9, size=16))
        cur = env.active.sum(axis=1)
        assert (cur <= prev).all()
        prev = cur
        assert (env.ship >= 0).all() and (env.ship <= env.cfg.world).all()


def test_fuel_penalty_applies_only_when_moving():
    env = SlickEnv(1, seed=0)
    env.pos[:] = 1e6  # nothing can be caught
    _, r_hold, *_ = env.step(np.array([8]))
    _, r_move, *_ = env.step(np.array([0]))
    assert r_hold[0] == 0 and r_move[0] < 0


def test_horizon_terminates():
    env = SlickEnv(2, Config(horizon=5))
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(np.zeros(2, dtype=int))
    assert done


def test_baselines_beat_random_by_a_wide_margin():
    cfg = REGIMES["train"]
    rand = run_episodes(BASELINES["random"], cfg, n_episodes=100)["recovered_mean"]
    greedy = run_episodes(BASELINES["greedy_nearest"], cfg, n_episodes=100)["recovered_mean"]
    assert greedy > rand + 0.3


def test_paired_evaluation_is_deterministic():
    cfg = REGIMES["train"]
    a = run_episodes(BASELINES["greedy_nearest"], cfg, n_episodes=100)["recovered_mean"]
    b = run_episodes(BASELINES["greedy_nearest"], cfg, n_episodes=100)["recovered_mean"]
    assert a == b


torch = pytest.importorskip("torch")


def test_gae_matches_hand_computation():
    from skimmer.ppo import gae

    r = torch.tensor([[1.0], [1.0]])
    v = torch.tensor([[0.5], [0.5]])
    adv, ret = gae(r, v, gamma=1.0, lam=1.0)
    # t=1: delta = 1 + 0 - 0.5 = 0.5 ; t=0: delta = 1 + 0.5 - 0.5 = 1.0 ; adv0 = 1.0 + 0.5 = 1.5
    assert adv[1].item() == pytest.approx(0.5) and adv[0].item() == pytest.approx(1.5)
    assert ret[0].item() == pytest.approx(2.0)


def test_policy_wrapper_returns_valid_actions():
    from skimmer.ppo import ActorCritic, policy_from_net

    pol = policy_from_net(ActorCritic())
    acts = pol(np.zeros((5, OBS_DIM), dtype=np.float32), None, Config())
    assert acts.shape == (5,) and ((acts >= 0) & (acts < 9)).all()
