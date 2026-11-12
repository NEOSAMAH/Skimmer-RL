# Explaining Skimmer-RL: a technical study guide

Use this to explain the project in an interview. Every number below is taken from the code.

## 1. The 30-second version

> "I built a simulated oil-spill recovery task and trained a reinforcement-learning agent on it. The agent is PPO that I implemented from scratch in PyTorch. It learned from random behaviour (2% oil recovered) to about 65%, which matches the best hand-written planner I built for comparison. I evaluated everything on the same 500 seeded episodes with confidence intervals, and on conditions it never trained on. The agent only clearly beat the baselines under strong currents, and it generalised worse than the simple heuristic. I also tried domain randomisation to fix that and it made things worse, so I documented that as a negative result."

## 2. The problem as an RL problem

| RL concept | In this project |
|---|---|
| Environment | 200 oil particles in a 100x100 world. A current moves them every step (speed 0.25-0.75 units/step, random direction per episode), the tide rotates the current direction by 0.01 rad/step, and diffusion adds noise (std 0.12). |
| Agent | A skimmer vessel moving 1.2 units/step. |
| Action (9, discrete) | 8 compass directions or hold. |
| Capture rule | Any particle within 2.5 units of the skimmer is removed. |
| Reward (per step) | `fraction of the 200 particles recovered this step - 0.0004 x distance moved`. Over an episode the recovery terms add up to the fraction recovered (max 1.0). |
| Episode | Exactly 200 steps, no early termination. |
| Observation (18 numbers) | Ship position (2), time left (1), oil left (1), oil density in 8 compass sectors within 30 units (8), relative centroid of remaining oil (2), relative nearest particle (2), current estimate (2). |

**Why these choices.** The observation is egocentric (relative to the ship) so the agent learns a rule like "oil is to my north-east" instead of memorising map coordinates. Time-left is included so the agent can reason about urgency, and it also makes the end-of-episode value genuinely zero (see GAE below).

## 3. The simulator (`skimmer/env.py`)

- **Vectorised**: all B environments are NumPy arrays of shape (B, N, 2). One `step()` advances all 64 training environments with no Python loop over environments. This is why 6.4 million environment steps take about 6 minutes on a CPU. RL is sample-hungry, so simulator speed is the bottleneck.
- **Reproducible**: each batch of environments draws from one seeded NumPy random generator, so the same seed gives the same set of slicks.
- **Regimes** (`REGIMES`): `train` (default), `strong_current` (0.7-1.0), `wide_slick` (std 9 instead of 5), `fast_tide` (0.04 rad/step). The non-train regimes are used only to test generalisation.

## 4. The baselines (`skimmer/baselines.py`)

All of them receive the same 18-number observation as the agent.

| Policy | Idea | Result (train regime) |
|---|---|---|
| random | uniform random action | 2.1% |
| chase_centroid | steer at the centre of the remaining oil | 36.8% |
| intercept | aim at where the centroid *will be*: time-to-reach t = distance / ship speed, target = centroid + current x t, iterated twice | 37.7% |
| greedy_nearest | steer at the nearest remaining particle | 64.4% |
| intercept_then_greedy | intercept from afar, switch to greedy within 12 units | 65.2% |

**The story to tell:** I expected the clever intercept planner to be strongest. It scored about 38% vs 64% for the simple greedy rule, because aiming at the centre makes the ship hover in the middle while the particles are spread around it. The hybrid fixed that. Lesson: test your assumptions about what a strong baseline is.

## 5. PPO in detail (`skimmer/ppo.py`)

**Network.** A shared body (18 -> 128 -> 128, tanh) with two heads: a policy head (9 action logits) and a value head (1 number). The policy head is initialised with a tiny orthogonal gain (0.01) so the starting policy is almost uniform, which gives early exploration.

**One training update**, in order:
1. **Rollout.** Run the *current* policy (sampling actions) in 64 environments for 200 steps. That is 12,800 transitions.
2. **Advantages (GAE).** For each step compute `delta_t = r_t + gamma x V(s_{t+1}) - V(s_t)` ("how much better than the value head expected"), then accumulate backwards: `A_t = delta_t + gamma x lambda x A_{t+1}`. With gamma = 0.99, lambda = 0.95. `lambda` trades bias against variance. The terminal value is set to 0: that is correct *because* time-left is in the observation.
3. **Returns.** `return_t = advantage_t + V(s_t)`, the target for the value head.
4. **Normalise advantages** across the batch (zero mean, unit std), which stabilises training.
5. **Optimise** for 4 epochs over shuffled minibatches of 2048 (7 minibatches per epoch). The loss has three terms:
   - **Policy (clipped):** `ratio = pi_new(a|s) / pi_old(a|s)`, loss = `-min(ratio x A, clip(ratio, 0.8, 1.2) x A)`. If an action was good (A > 0) we raise its probability, but never by more than 20% in one update. That limit is what makes it *proximal*.
   - **Value:** squared error between the value head and the returns (weight 0.5).
   - **Entropy bonus:** `-0.01 x entropy`, which keeps the policy from becoming deterministic too early.
6. Gradient-norm clipping at 0.5. Adam (lr 3e-4, eps 1e-5), with the learning rate linearly annealed to 0 over training.

500 updates x 12,800 samples = **6.4 million environment steps** per run. At evaluation the agent acts **greedily** (the most probable action) instead of sampling.

## 6. Evaluation methodology (`skimmer/runner.py`, `skimmer/evaluate.py`)

- **Fixed seeds, paired comparison.** Every policy plays the same 500 slicks (seeds 12345 + batch offset, in batches of 100). So differences between policies are not caused by easier or harder random draws.
- **95% confidence interval** = `1.96 x std / sqrt(n)` over the 500 episodes.
- **Paired difference.** For "PPO vs best baseline" I subtract the per-episode results on the same slicks, then compute the CI of the *difference*. That is much tighter than comparing two separate averages. For example, "+0.3 ± 1.0" means the difference is indistinguishable from zero.
- **Two seeds per RL variant.** RL results can swing a lot between random seeds, so a single run is not evidence. (Two is the minimum, not a lot. More would be better.)
- **Held-out regimes** test generalisation. The plain agent trained only on `train`; the domain-randomised agent trained on `train`, `strong_current` and `wide_slick`, with `fast_tide` held out.

## 7. Results and the honest reading

| Regime | Best baseline | PPO (plain) | PPO + domain randomisation |
|---|---|---|---|
| train | 65.2 ± 2.9 | 65.4 / 64.1 | 50.9 / 50.1 |
| strong_current | 24.0 (intercept) | 25.4 / 26.3 | 25.7 / 25.9 |
| wide_slick | 44.4 | 39.8 / 39.3 | 30.1 / 30.4 |
| fast_tide | 96.9 | 86.8 / 86.4 | 75.8 / 71.0 |

1. PPO learned a strong policy from scratch (2% to 65%) and **ties** the best baseline on training conditions (paired difference +0.3 ± 1.0).
2. It beats the best baseline **only under strong currents** (+2.3 ± 1.2 points, paired). That is a modest edge.
3. It is **worse on unseen conditions** (-4.6 on wide slicks, -10.1 on fast tides). A heuristic has nothing learned that could go out-of-distribution.
4. **Domain randomisation hurt.** Hypotheses I have *not* tested: too few updates for a harder mixture (final entropy stayed at about 1.15 vs about 0.9), and reward scales that differ about 3x across regimes make advantage estimates noisy.
5. A further untested hypothesis for the tie: the task may be simple enough that a greedy rule is already near-optimal, leaving little for RL to improve on.

## 8. Limitations to state yourself before they ask

- **Idealised sensing**: the agent gets the exact nearest-particle position and the current velocity. A real vessel would see noisy radar or camera data.
- **Simulator only**: nothing was validated against real spill data or vessel dynamics. The physics is a toy (constant-speed ship, simple drift).
- **Small experiment**: two seeds, one network size, no hyperparameter search.
- **Serving**: the FastAPI endpoint is tested locally, but the Docker image was never built or run.

## 9. Likely interview questions, with answers

**Why PPO?** It is stable, simple and a standard default for discrete-action problems. Clipping prevents one noisy batch from destroying the policy. I implemented it myself to understand each piece.

**Why from scratch instead of Stable-Baselines3?** To understand and be able to defend every line. In production I would use a maintained library.

**Why a baseline?** Without one, "reward went up" proves nothing. Mine showed that RL only matched a simple heuristic, which is a more useful and honest finding.

**Why did RL not beat greedy?** I don't know for certain. Hypotheses: the task is easy enough that greedy is near-optimal, or the agent is undertrained. Next I would test the second by training longer, and the first by making the task harder (obstacles, a fuel constraint, noisy sensing).

**What is GAE for?** It estimates how good each action was, balancing bias and variance with lambda. Without it the agent cannot tell which of the 200 steps caused the final outcome.

**Why vectorised NumPy?** RL needs millions of steps. Stepping 64 environments in one array operation makes 6.4M steps feasible in minutes on a CPU.

**How would you deploy this?** Export the policy to ONNX, quantise, and measure CPU latency, since vessels have limited compute. I have not done the ONNX step, only a PyTorch FastAPI endpoint.

**What would you do with more time?** Fix domain randomisation (longer training, per-regime reward normalisation, a curriculum), add sensor noise with a recurrent policy, and compare with search-based planning (MCTS).

**What did you learn from the failures?** That my intuition about the strongest baseline was wrong, and that a more varied training mix does not automatically generalise better.

## 10. What not to say

- Don't say it "beats classical planning". It ties on training conditions and loses on unseen ones.
- Don't say it works on real spills. It is a toy simulator.
- Don't blame the domain-randomisation failure on a specific cause. You don't know it yet.

## 11. Where to find things

| What | File |
|---|---|
| Simulator, regimes, observation layout | `skimmer/env.py` |
| Hand-written planners | `skimmer/baselines.py` |
| PPO, GAE, network, training loop | `skimmer/ppo.py` |
| Paired evaluation | `skimmer/runner.py`, `skimmer/evaluate.py` |
| Serving API | `skimmer/serve.py` |
| Tests (11) | `tests/` |
| Results, plot, trained weights | `results/` |
