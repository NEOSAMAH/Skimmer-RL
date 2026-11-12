"""Animate one episode: trained PPO agent vs. a hand-written baseline on the *same* slick.

    python -m skimmer.visualize --policy results/policy_seed0.pt --regime strong_current --seed 3
Writes results/episode.gif (or --out).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from .baselines import intercept_then_greedy
from .env import REGIMES, SlickEnv
from .ppo import ActorCritic, policy_from_net


def rollout(policy, cfg, seed):
    env = SlickEnv(1, cfg, seed=seed)
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    frames = []
    done = False
    while not done:
        frames.append((env.pos[0].copy(), env.active[0].copy(), env.ship[0].copy()))
        obs, _, done, info = env.step(policy(obs, rng, cfg))
    frames.append((env.pos[0].copy(), env.active[0].copy(), env.ship[0].copy()))
    return frames, float(info["recovered_frac"][0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="results/policy_seed0.pt")
    p.add_argument("--regime", default="train", choices=list(REGIMES))
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="results/episode.gif")
    p.add_argument("--stride", type=int, default=2, help="render every Nth step")
    a = p.parse_args()

    cfg = REGIMES[a.regime]
    net = ActorCritic()
    net.load_state_dict(torch.load(a.policy))
    net.eval()
    runs = {
        "PPO agent": rollout(policy_from_net(net), cfg, a.seed),
        "intercept→greedy baseline": rollout(intercept_then_greedy, cfg, a.seed),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    n = len(next(iter(runs.values()))[0])
    artists = []
    for ax, (name, (frames, final)) in zip(axes, runs.items()):
        ax.set_xlim(0, cfg.world)
        ax.set_ylim(0, cfg.world)
        ax.set_aspect("equal")
        ax.set_facecolor("#dbeaf5")
        ax.set_xticks([])
        ax.set_yticks([])
        oil = ax.scatter([], [], s=6, c="#1b1b1b")
        ship = ax.scatter([], [], s=70, c="#d62728", marker="^", zorder=3)
        trail, = ax.plot([], [], c="#d62728", lw=0.8, alpha=0.6)
        title = ax.set_title(name)
        artists.append((ax, name, frames, final, oil, ship, trail, title))

    def update(i):
        k = min(i * a.stride, n - 1)
        out = []
        for ax, name, frames, final, oil, ship, trail, title in artists:
            pos, active, s = frames[k]
            oil.set_offsets(pos[active])
            ship.set_offsets(s[None])
            path = np.array([f[2] for f in frames[: k + 1]])
            trail.set_data(path[:, 0], path[:, 1])
            rec = 1 - active.mean()
            title.set_text(f"{name}\nstep {k}/{n - 1}   recovered {rec:.0%}")
            out += [oil, ship, trail, title]
        return out

    fig.suptitle(f"regime: {a.regime}, seed {a.seed}", fontsize=10)
    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=(n + a.stride - 1) // a.stride, blit=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=15))
    for name, (_, final) in runs.items():
        print(f"{name}: {final:.1%} recovered")
    print("saved", a.out)


if __name__ == "__main__":
    main()
