"""Run any policy for full episodes and summarise with confidence intervals."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .env import Config, SlickEnv

Policy = Callable[[np.ndarray, np.random.Generator, Config], np.ndarray]


def run_episodes(policy: Policy, cfg: Config, n_episodes: int = 500, seed: int = 12345, batch: int = 100) -> dict:
    """Fixed evaluation seeds => every policy sees the *same* 500 slicks (paired comparison)."""
    recovered, distance = [], []
    for start in range(0, n_episodes, batch):
        env = SlickEnv(batch, cfg, seed=seed + start)
        obs = env.reset(seed=seed + start)
        rng = np.random.default_rng(seed + start + 1)
        done, info = False, {}
        while not done:
            obs, _, done, info = env.step(policy(obs, rng, cfg))
        recovered.append(info["recovered_frac"])
        distance.append(info["distance"])
    rec, dist = np.concatenate(recovered)[:n_episodes], np.concatenate(distance)[:n_episodes]
    return {
        "recovered_mean": float(rec.mean()),
        "recovered_ci95": float(1.96 * rec.std(ddof=1) / np.sqrt(len(rec))),
        "recovered_per_episode": rec,
        "distance_mean": float(dist.mean()),
    }
