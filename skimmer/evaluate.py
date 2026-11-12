"""Evaluate baselines and trained policies on identical seeds, across regimes.

Usage: python -m skimmer.evaluate --policies results/policy_seed0.pt results/policy_seed1.pt
Writes results/results.json, results/results.md and results/comparison.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import BASELINES
from .env import REGIMES
from .runner import run_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=[])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    contenders = dict(BASELINES)
    if a.policies:
        import torch

        from .ppo import ActorCritic, policy_from_net

        for p in a.policies:
            net = ActorCritic()
            net.load_state_dict(torch.load(p))
            net.eval()
            contenders[f"ppo[{Path(p).stem.replace('policy_', '')}]"] = policy_from_net(net)

    results: dict = {}
    for regime, cfg in REGIMES.items():
        results[regime] = {}
        for name, pol in contenders.items():
            r = run_episodes(pol, cfg, n_episodes=a.episodes)
            results[regime][name] = {k: v for k, v in r.items() if k != "recovered_per_episode"}
            results[regime][name]["_eps"] = r["recovered_per_episode"].tolist()

    # paired comparison: PPO vs best baseline on the very same slicks => much tighter than unpaired CIs
    best_base = lambda reg: max((n for n in BASELINES if n != "random"), key=lambda n: results[reg][n]["recovered_mean"])
    lines = ["| Regime | Policy | Oil recovered (mean ± 95% CI) | Distance |", "|---|---|---|---|"]
    for regime in REGIMES:
        for name, r in results[regime].items():
            lines.append(f"| {regime} | {name} | {r['recovered_mean']*100:.1f}% ± {r['recovered_ci95']*100:.1f} | {r['distance_mean']:.0f} |")
    lines += ["", "Paired difference, best PPO seed vs best baseline (same 500 slicks):", ""]
    for regime in REGIMES:
        ppos = [n for n in results[regime] if n.startswith("ppo")]
        if not ppos:
            continue
        bp = max(ppos, key=lambda n: results[regime][n]["recovered_mean"])
        diff = np.array(results[regime][bp]["_eps"]) - np.array(results[regime][best_base(regime)]["_eps"])
        lines.append(f"- {regime}: {bp} minus {best_base(regime)} = {diff.mean()*100:+.1f} pts ± {1.96*diff.std(ddof=1)/np.sqrt(len(diff))*100:.1f}")
    (out / "results.md").write_text("\n".join(lines))
    for reg in results:
        for n in results[reg]:
            results[reg][n].pop("_eps")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(next(iter(results.values())).keys())
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(len(results)) + i * w, [results[r][n]["recovered_mean"] * 100 for r in results], w,
               yerr=[results[r][n]["recovered_ci95"] * 100 for r in results], label=n, capsize=2)
    ax.set_xticks(np.arange(len(results)) + 0.4 - w / 2, list(results))
    ax.set_ylabel("oil recovered (%)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Oil recovered by policy and regime (error bars: 95% CI)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)


if __name__ == "__main__":
    main()
