"""PPO written from scratch in PyTorch (no RL library) so every line is explainable.

The five ideas, in the order they appear below:
  1. Actor-critic: one network proposes actions (policy), one estimates how good a state is (value).
  2. Roll out the current policy in many envs at once.
  3. Advantage = "how much better was this action than the value network expected?" (GAE).
  4. Update the policy to make good actions more likely ...
  5. ... but clip the update so one noisy batch cannot wreck the policy (the "P" in PPO).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import Config, N_ACTIONS, OBS_DIM, REGIMES, SlickEnv


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, N_ACTIONS)   # action logits
        self.v = nn.Linear(hidden, 1)            # state value
        # small policy init => near-uniform start => lots of exploration
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return Categorical(logits=self.pi(h)), self.v(h).squeeze(-1)


def gae(rewards, values, gamma: float, lam: float):
    """Generalised Advantage Estimation. rewards/values: (T, B); terminal value is 0 because
    the observation contains time-remaining, so the value at the time limit really is 0."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[1])
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else torch.zeros(rewards.shape[1])
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


DR_REGIMES = ("train", "strong_current", "wide_slick")  # fast_tide is deliberately held out


def train(updates: int = 300, n_envs: int = 64, seed: int = 0, out: Path = Path("results"), log_every: int = 10, randomize: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = Config()
    env = SlickEnv(n_envs, cfg, seed=seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    gamma, lam, clip, ent_coef, vf_coef = 0.99, 0.95, 0.2, 0.01, 0.5
    T = cfg.horizon
    history = []
    t0 = time.time()

    for u in range(1, updates + 1):
        # linearly anneal the learning rate: big steps early, careful steps late
        for g in opt.param_groups:
            g["lr"] = 3e-4 * (1 - (u - 1) / updates)

        # ---- 2. rollout one full episode in every env
        obs_buf = torch.zeros(T, n_envs, OBS_DIM)
        act_buf = torch.zeros(T, n_envs, dtype=torch.long)
        logp_buf, val_buf, rew_buf = (torch.zeros(T, n_envs) for _ in range(3))
        if randomize:  # domain randomisation: a different physical regime each update
            env.cfg = REGIMES[DR_REGIMES[np.random.randint(len(DR_REGIMES))]]
        obs = torch.as_tensor(env.reset())
        for t in range(T):
            with torch.no_grad():
                dist, value = net(obs)
                action = dist.sample()
            nxt, reward, _, info = env.step(action.numpy())
            obs_buf[t], act_buf[t], val_buf[t] = obs, action, value
            logp_buf[t], rew_buf[t] = dist.log_prob(action), torch.as_tensor(reward, dtype=torch.float32)
            obs = torch.as_tensor(nxt)

        # ---- 3. advantages and returns
        adv, ret = gae(rew_buf, val_buf, gamma, lam)

        # flatten (T, B) -> (T*B)
        b_obs, b_act = obs_buf.reshape(-1, OBS_DIM), act_buf.reshape(-1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)   # normalise: stabilises training

        # ---- 4/5. several epochs of clipped policy-gradient updates on minibatches
        idx = np.arange(T * n_envs)
        for _ in range(4):
            np.random.shuffle(idx)
            for s in range(0, len(idx), 2048):
                mb = torch.as_tensor(idx[s : s + 2048])
                dist, value = net(b_obs[mb])
                ratio = torch.exp(dist.log_prob(b_act[mb]) - b_logp[mb])      # new prob / old prob
                surr = torch.min(ratio * b_adv[mb], torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv[mb])
                loss = -surr.mean() + vf_coef * (value - b_ret[mb]).pow(2).mean() - ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        rec = float(info["recovered_frac"].mean())
        history.append({"update": u, "train_recovered": rec, "entropy": float(dist.entropy().mean().item())})
        if u % log_every == 0 or u == 1:
            print(f"update {u:4d}/{updates}  recovered={rec:.3f}  entropy={history[-1]['entropy']:.2f}  {time.time()-t0:.0f}s", flush=True)

    out.mkdir(exist_ok=True)
    tag = "dr_" if randomize else ""
    torch.save(net.state_dict(), out / f"policy_{tag}seed{seed}.pt")
    (out / f"train_log_{tag}seed{seed}.json").write_text(json.dumps(history))
    return net


def policy_from_net(net: ActorCritic, greedy: bool = True):
    """Wrap a trained network as a `policy(obs, rng, cfg) -> actions` for the shared runner."""

    def policy(obs, rng, cfg):
        with torch.no_grad():
            dist, _ = net(torch.as_tensor(obs))
            return (dist.probs.argmax(-1) if greedy else dist.sample()).numpy()

    return policy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="domain randomisation over several regimes")
    a = ap.parse_args()
    train(updates=a.updates, seed=a.seed, randomize=a.randomize)
