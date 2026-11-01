# Gamebot v0.3.2 PPO pilot

Training seeds: 3JY1 6FLR, 7W3L P1Z9, SWGF G7WV. Held-out seeds: WE4Y Y4DM, F2HZ 6ZVD. Test rooms: gurdyJr, larryJr, dukeOfFlies, theBloat.
Six PPO updates, 1536 decisions / 6138 logic frames, action repeat 4, episode frame limit 600.

## Fixed test set

| Policy | Episodes | Clear rate | Death rate | Mean damage dealt | Mean damage taken | Damage-bearing fire decision rate* | Mean clear frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 8 | 0.000 | 0.250 | 31.875 | 4.375 | 0.058 | — |
| Untrained Actor | 8 | 0.000 | 0.375 | 29.375 | 4.750 | 0.063 | — |
| Final PPO | 8 | 0.000 | 0.500 | 34.188 | 4.750 | 0.060 | — |
| Best held-out PPO | 8 | 0.000 | 0.375 | 36.188 | 4.375 | 0.068 | — |

## Training and held-out metrics

| Round | Curriculum stage | Episode reward | Actor loss | Critic loss | Entropy | Approx KL | Clip fraction | Explained variance | Test clear rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 静止目标瞄准近似 | -9.734 | -0.017 | 8.712 | 3.496 | 0.008 | 0.009 | 0.009 | 0.000 |
| 2 | 移动目标瞄准 | 31.400 | -0.009 | 3.480 | 3.485 | -0.001 | 0.002 | -0.042 | 0.000 |
| 3 | 弹幕躲避 | -38.092 | -0.005 | 8.421 | 3.527 | 0.001 | 0.000 | -0.026 | 0.000 |
| 4 | 多敌人组合 | 16.775 | -0.007 | 9.236 | 3.464 | 0.001 | 0.000 | -0.115 | 0.000 |
| 5 | Boss 战斗 | -15.945 | -0.003 | 15.309 | 3.525 | -0.000 | 0.000 | -0.016 | 0.000 |
| 6 | 混合场景 | 3.990 | -0.011 | 5.097 | 3.488 | -0.001 | 0.001 | -0.005 | 0.000 |

## Per-room held-out comparison

| Room | Policy | Episodes | Clear rate | Death rate | Damage dealt | Damage taken | Fire-damage decision proxy |
|---|---|---:|---:|---:|---:|---:|---:|
| gurdyJr | random | 2 | 0.000 | 0.000 | 33.500 | 4.000 | 0.046 |
| gurdyJr | untrainedActor | 2 | 0.000 | 0.500 | 24.250 | 4.000 | 0.048 |
| gurdyJr | finalPPO | 2 | 0.000 | 0.000 | 44.500 | 3.000 | 0.072 |
| gurdyJr | bestHeldoutPPO | 2 | 0.000 | 0.500 | 32.000 | 5.000 | 0.046 |
| larryJr | random | 2 | 0.000 | 0.000 | 29.750 | 3.000 | 0.062 |
| larryJr | untrainedActor | 2 | 0.000 | 0.000 | 22.750 | 5.000 | 0.045 |
| larryJr | finalPPO | 2 | 0.000 | 0.000 | 15.750 | 4.000 | 0.032 |
| larryJr | bestHeldoutPPO | 2 | 0.000 | 0.500 | 36.750 | 4.000 | 0.082 |
| dukeOfFlies | random | 2 | 0.000 | 0.000 | 59.000 | 4.500 | 0.090 |
| dukeOfFlies | untrainedActor | 2 | 0.000 | 0.000 | 65.250 | 4.000 | 0.118 |
| dukeOfFlies | finalPPO | 2 | 0.000 | 1.000 | 64.250 | 6.000 | 0.109 |
| dukeOfFlies | bestHeldoutPPO | 2 | 0.000 | 0.000 | 60.250 | 3.500 | 0.100 |
| theBloat | random | 2 | 0.000 | 1.000 | 5.250 | 6.000 | 0.021 |
| theBloat | untrainedActor | 2 | 0.000 | 1.000 | 5.250 | 6.000 | 0.022 |
| theBloat | finalPPO | 2 | 0.000 | 1.000 | 12.250 | 6.000 | 0.030 |
| theBloat | bestHeldoutPPO | 2 | 0.000 | 0.500 | 15.750 | 5.000 | 0.039 |

## Interpretation

- No policy cleared a room on the fixed test set; mean clear time is therefore unavailable.
- Final PPO did not beat Random on death rate or mean episode reward. Best-held-out PPO improved mean damage and the action-level damage proxy modestly, but still cleared no room.
- Critic explained variance stayed near zero or below zero. Entropy remained high and approximate KL/clip fraction stayed small across six updates.
- Do not expand training from this result. Improve value learning and combat signal, then rerun the same held-out suite before claiming progress.
- *The public environment exposes enemy HP deltas, not projectile-hit events. This rate is the fraction of fire decision intervals that coincide with enemy damage, not per-shot accuracy.
- Gurdy Jr. is only a stationary-target approximation; the existing environment has no standalone stationary-target or isolated-dodge scenario.
