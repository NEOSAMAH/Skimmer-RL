| Regime | Policy | Oil recovered (mean ± 95% CI) | Distance |
|---|---|---|---|
| train | random | 2.1% ± 0.7 | 207 |
| train | greedy_nearest | 64.4% ± 2.9 | 170 |
| train | chase_centroid | 36.8% ± 1.2 | 167 |
| train | intercept | 37.7% ± 1.2 | 174 |
| train | intercept_then_greedy | 65.2% ± 2.9 | 177 |
| train | ppo[seed0] | 65.4% ± 2.6 | 172 |
| train | ppo[seed1] | 64.1% ± 2.5 | 169 |
| train | ppo[dr_seed0] | 50.9% ± 1.9 | 166 |
| train | ppo[dr_seed1] | 50.1% ± 1.8 | 165 |
| strong_current | random | 2.2% ± 0.7 | 207 |
| strong_current | greedy_nearest | 17.4% ± 1.7 | 127 |
| strong_current | chase_centroid | 19.9% ± 1.6 | 119 |
| strong_current | intercept | 24.0% ± 1.7 | 128 |
| strong_current | intercept_then_greedy | 20.0% ± 1.8 | 138 |
| strong_current | ppo[seed0] | 25.4% ± 2.4 | 119 |
| strong_current | ppo[seed1] | 26.3% ± 2.4 | 119 |
| strong_current | ppo[dr_seed0] | 25.7% ± 2.2 | 116 |
| strong_current | ppo[dr_seed1] | 25.9% ± 2.1 | 116 |
| wide_slick | random | 2.3% ± 0.6 | 207 |
| wide_slick | greedy_nearest | 44.0% ± 2.1 | 186 |
| wide_slick | chase_centroid | 20.2% ± 0.6 | 168 |
| wide_slick | intercept | 20.8% ± 0.6 | 174 |
| wide_slick | intercept_then_greedy | 44.4% ± 2.1 | 192 |
| wide_slick | ppo[seed0] | 39.8% ± 1.7 | 172 |
| wide_slick | ppo[seed1] | 39.3% ± 1.6 | 170 |
| wide_slick | ppo[dr_seed0] | 30.1% ± 1.0 | 165 |
| wide_slick | ppo[dr_seed1] | 30.4% ± 1.0 | 163 |
| fast_tide | random | 0.8% ± 0.4 | 207 |
| fast_tide | greedy_nearest | 96.6% ± 0.5 | 232 |
| fast_tide | chase_centroid | 39.9% ± 0.3 | 240 |
| fast_tide | intercept | 39.8% ± 0.3 | 240 |
| fast_tide | intercept_then_greedy | 96.9% ± 0.4 | 232 |
| fast_tide | ppo[seed0] | 86.8% ± 0.5 | 240 |
| fast_tide | ppo[seed1] | 86.4% ± 0.5 | 240 |
| fast_tide | ppo[dr_seed0] | 75.8% ± 0.7 | 239 |
| fast_tide | ppo[dr_seed1] | 71.0% ± 0.8 | 239 |

Paired difference, best PPO seed vs best baseline (same 500 slicks):

- train: ppo[seed0] minus intercept_then_greedy = +0.3 pts ± 1.0
- strong_current: ppo[seed1] minus intercept = +2.3 pts ± 1.2
- wide_slick: ppo[seed0] minus intercept_then_greedy = -4.6 pts ± 1.2
- fast_tide: ppo[seed0] minus intercept_then_greedy = -10.1 pts ± 0.5