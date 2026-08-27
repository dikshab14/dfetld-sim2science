| method | simulator calls | wall-clock (s) | notes |
|---|---|---|---|
| DF-ETLD (ours) | 1,500,000 | 169 | M=50, J=20, 1500 steps; tame=0.08; no simulator derivatives |
| DF-ETLD @ 10k-call budget | 10,000 | 8 | small-budget run: 10,000 simulator calls |
| SMC-ABC (6 summaries) | 88,000 | 577 | run to convergence (tolerance plateau); pop=200, m=20 |
| adSGLD-KS (FD-gradient proxy) | 6,144,000 | 2574 | 4 chains x 16000 steps, J=24; FD gradients; KERNEL score; beta=10; RUN TO CONVERGENCE (split-Rhat 1.015). The earlier single 2000-step chain (192,000 calls) was NOT converged and gave an optimistic 0.066 vs 0.081 |
| adSGLD-ES (FD-gradient proxy) | 6,144,000 | 2759 | 4 chains x 16000 steps, J=24; FD gradients; ENERGY score (scale-free: no bandwidth); beta=4.38 by Bayes-factor matching; RUN TO CONVERGENCE (split-Rhat 1.012) |
