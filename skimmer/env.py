"""Vectorised oil-slick recovery simulation.

An oil slick is a cloud of N particles that drifts with an (unknown-per-episode) ocean
current, slowly rotates with the tide, and spreads by random diffusion. A skimmer vessel
removes every particle that comes within `capture_radius` of it. The goal is to recover as
much oil as possible in `horizon` steps while spending little fuel.

All B environments are stepped at once with NumPy arrays (no Python loop over envs), which
is what makes training RL in minutes on a CPU possible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# 8 compass directions + "hold position"
_DIRS = np.array(
    [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [0, 0]], dtype=np.float64
)
_DIRS[:8] /= np.linalg.norm(_DIRS[:8], axis=1, keepdims=True)
N_ACTIONS = 9
OBS_DIM = 18


@dataclass(frozen=True)
class Config:
    world: float = 100.0
    n_particles: int = 200
    horizon: int = 200
    skimmer_speed: float = 1.2
    capture_radius: float = 2.5
    slick_sigma: float = 5.0
    current_min: float = 0.25
    current_max: float = 0.75
    tide_rate: float = 0.01        # radians per step the current direction rotates
    diffusion: float = 0.12
    fuel_cost: float = 0.0004      # reward penalty per unit distance travelled
    sense_radius: float = 30.0     # radius of the egocentric density sensor


# Evaluation-time regimes. "train" matches training; the others test generalisation.
REGIMES = {
    "train": Config(),
    "strong_current": replace(Config(), current_min=0.7, current_max=1.0),
    "wide_slick": replace(Config(), slick_sigma=9.0),
    "fast_tide": replace(Config(), tide_rate=0.04),
}


class SlickEnv:
    def __init__(self, n_envs: int, cfg: Config = Config(), seed: int = 0):
        self.B, self.cfg = n_envs, cfg
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ dynamics
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, B, N = self.cfg, self.B, self.cfg.n_particles
        centre = self.rng.uniform(0.3 * c.world, 0.7 * c.world, size=(B, 1, 2))
        self.pos = centre + self.rng.normal(0, c.slick_sigma, size=(B, N, 2))
        self.active = np.ones((B, N), dtype=bool)
        speed = self.rng.uniform(c.current_min, c.current_max, size=B)
        self.theta = self.rng.uniform(0, 2 * np.pi, size=B)
        self.speed = speed
        self.tide_dir = self.rng.choice([-1.0, 1.0], size=B)
        # skimmer starts on a random point of the world border
        edge = self.rng.uniform(0, c.world, size=B)
        side = self.rng.integers(0, 4, size=B)
        sx = np.where(side == 0, 0, np.where(side == 1, c.world, edge))
        sy = np.where(side == 2, 0, np.where(side == 3, c.world, edge))
        self.ship = np.stack([sx, sy], axis=1)
        self.t = 0
        self.distance = np.zeros(B)
        return self.observe()

    def current(self) -> np.ndarray:
        return self.speed[:, None] * np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)

    def step(self, actions: np.ndarray):
        c = self.cfg
        move = _DIRS[actions] * c.skimmer_speed
        new_ship = np.clip(self.ship + move, 0, c.world)
        travelled = np.linalg.norm(new_ship - self.ship, axis=1)
        self.ship = new_ship
        self.distance += travelled

        # slick drifts, spreads; the tide rotates the current
        self.theta += self.tide_dir * c.tide_rate
        self.pos += self.current()[:, None, :] + self.rng.normal(0, c.diffusion, size=self.pos.shape)

        # recovery
        d = np.linalg.norm(self.pos - self.ship[:, None, :], axis=2)
        caught = self.active & (d < c.capture_radius)
        self.active &= ~caught
        recovered = caught.sum(axis=1) / c.n_particles

        reward = recovered - c.fuel_cost * travelled
        self.t += 1
        done = self.t >= c.horizon
        info = {"recovered_frac": 1.0 - self.active.sum(axis=1) / c.n_particles, "distance": self.distance.copy()}
        return self.observe(), reward, done, info

    # --------------------------------------------------------------- observation
    def observe(self) -> np.ndarray:
        """18-d, egocentric, roughly in [-1, 1]: what a real vessel's sensors could provide.

        [0:2]    ship position (scaled to [-1, 1])
        [2]      time remaining (fraction)
        [3]      oil remaining (fraction)
        [4:12]   oil density in 8 compass sectors within sense_radius
        [12:14]  centroid of remaining oil, relative to ship
        [14:16]  nearest oil particle, relative to ship
        [16:18]  current estimate (vx, vy), scaled by max current
        """
        c = self.cfg
        rel = self.pos - self.ship[:, None, :]                     # (B,N,2)
        dist = np.linalg.norm(rel, axis=2)
        act = self.active
        n_act = act.sum(axis=1)
        remaining = n_act / c.n_particles

        # 8 sector densities within the sensor radius
        ang = (np.arctan2(rel[..., 1], rel[..., 0]) + 2 * np.pi) % (2 * np.pi)
        sector = np.floor(ang / (np.pi / 4)).astype(int) % 8
        near = act & (dist < c.sense_radius)
        sect = np.zeros((self.B, 8))
        for k in range(8):
            sect[:, k] = (near & (sector == k)).sum(axis=1)
        sect /= 0.25 * c.n_particles

        denom = np.maximum(n_act, 1)[:, None]
        centroid = (rel * act[..., None]).sum(axis=1) / denom       # relative to ship
        masked = np.where(act, dist, np.inf)
        idx = masked.argmin(axis=1)
        nearest = rel[np.arange(self.B), idx]
        nearest = np.where(np.isfinite(masked.min(axis=1))[:, None], nearest, 0.0)

        obs = np.concatenate(
            [
                self.ship / c.world * 2 - 1,                                   # 0:2
                np.full((self.B, 1), 1 - self.t / c.horizon),                   # 2
                remaining[:, None],                                             # 3
                sect,                                                           # 4:12
                np.clip(centroid / c.world, -1, 1),                             # 12:14
                np.clip(nearest / c.world, -1, 1),                              # 14:16
                self.current() / c.current_max,                                 # 16:18
            ],
            axis=1,
        )
        assert obs.shape[1] == OBS_DIM
        return obs.astype(np.float32)
