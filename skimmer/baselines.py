"""Hand-written policies. They see exactly the same 18-d observation as the RL agent.

Why have them? An RL result means nothing without a baseline: "reward went up" is not
evidence of anything, "it beats a sensible hand-written planner by X +/- Y" is.
"""
from __future__ import annotations

import numpy as np

from .env import _DIRS, Config, N_ACTIONS

# observation slices (see SlickEnv.observe)
CENTROID = slice(12, 14)
NEAREST = slice(14, 16)
CURRENT = slice(16, 18)


def _toward(vec: np.ndarray) -> np.ndarray:
    """Pick the compass action whose direction is closest to `vec`; hold if vec ~ 0."""
    scores = vec @ _DIRS[:8].T
    act = scores.argmax(axis=1)
    return np.where(np.linalg.norm(vec, axis=1) < 1e-6, 8, act)


def random_policy(obs: np.ndarray, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    return rng.integers(0, N_ACTIONS, size=len(obs))


def greedy_nearest(obs: np.ndarray, rng, cfg: Config) -> np.ndarray:
    """Always steer at the closest remaining particle."""
    return _toward(obs[:, NEAREST])


def chase_centroid(obs: np.ndarray, rng, cfg: Config) -> np.ndarray:
    """Steer at the centre of the remaining slick (ignores that it is moving)."""
    return _toward(obs[:, CENTROID])


def intercept(obs: np.ndarray, rng, cfg: Config, current_max: float = 0.75) -> np.ndarray:
    """Aim where the centroid *will be* (lead pursuit), not where it is now.

    Estimate time-to-reach t = |centroid| / ship_speed, then aim at centroid + current * t.
    Iterating twice handles the fact that t itself depends on the aim point.
    Weakness (measured, see results): it gets the ship to the slick quickly but then hovers
    near the centroid instead of sweeping through the spread-out particles.
    """
    centroid = obs[:, CENTROID] * cfg.world
    drift = obs[:, CURRENT] * current_max  # undo the observation scaling
    target = centroid
    for _ in range(2):
        t = np.linalg.norm(target, axis=1, keepdims=True) / cfg.skimmer_speed
        target = centroid + drift * t
    return _toward(target)


def intercept_then_greedy(obs: np.ndarray, rng, cfg: Config, switch_dist: float = 12.0) -> np.ndarray:
    """Hybrid: lead-pursuit the slick from afar, then eat the nearest oil once close."""
    near_dist = np.linalg.norm(obs[:, NEAREST] * cfg.world, axis=1)
    return np.where(near_dist < switch_dist, greedy_nearest(obs, rng, cfg), intercept(obs, rng, cfg))


BASELINES = {
    "random": random_policy,
    "greedy_nearest": greedy_nearest,
    "chase_centroid": chase_centroid,
    "intercept": intercept,
    "intercept_then_greedy": intercept_then_greedy,
}
