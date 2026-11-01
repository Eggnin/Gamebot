# PPO continuation (iterations 7–10)

Resumed from iteration 6; 1024 additional decisions, action repeat 4, frame limit 600. Training seeds: 3JY1 6FLR, 7W3L P1Z9, SWGF G7WV. Held-out seeds: WE4Y Y4DM, F2HZ 6ZVD. Test set: Gurdy Jr., Larry Jr., Duke of Flies, The Bloat (8 episodes per checkpoint).

| Iteration | Episode reward | Actor loss | Critic loss | Entropy | Explained variance | Test clear rate | Test death rate | Test damage dealt | Test damage taken |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 (start) | — | — | — | — | — | 0.000 | 0.125 | 33.562 | 4.125 |
| 7 | -41.561 | -0.002 | 11.545 | 3.520 | 0.044 | 0.000 | 0.125 | 30.938 | 3.375 |
| 8 | 14.761 | -0.004 | 11.023 | 3.515 | -0.025 | 0.000 | 0.250 | 35.250 | 4.625 |
| 9 | 21.424 | -0.007 | 8.003 | 3.537 | 0.007 | 0.000 | 0.625 | 38.625 | 5.000 |
| 10 | -22.010 | -0.003 | 10.926 | 3.557 | 0.014 | 0.000 | 0.250 | 29.250 | 4.125 |

Best held-out checkpoint: iteration 6 (continuation-start); clear rate 0.000, death rate 0.125, mean damage dealt 33.562, damage taken 4.125.

The best candidate remained the iteration-6 starting checkpoint. Iteration 9 increased mean damage to 38.625 but raised held-out death rate to 62.5%, so it was not selected. No checkpoint cleared a test room. Critic explained variance remained near zero. This continuation does not justify scaling training yet.
