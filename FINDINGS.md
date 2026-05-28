# Prior / cond / kernel-form sweep — findings

All small-grid runs are 8×8×3 = 192 dim, YOGI, blks=4, bs=32, 30k steps.

## Experiment 1: ls prior shape (Makkunda kernel, 2-D cond, seed 19)

| ls prior | gMLP MSE | FM MSE | comment |
|---|---|---|---|
| Uniform-on-log[1, 100] (narrow) | 0.008 | 0.18 | branch baseline |
| Uniform-on-log[0.1, 1000] (wide, 4 OOMs) | 0.015 | 0.19 | range doesn't matter |
| Beta(4, 1) + LogScale[1, 100] | 0.013 | **1.47** | piles mass at ls=100 → FM collapses |
| LogNormal(log 20, 1) | 0.010 | 0.19 | bell in log — fine |

**Takeaway**: FM fails when the ls prior concentrates near a boundary
(Beta(4,1) puts ~80% of mass within a factor of 2 of the upper bound).
Wide-but-flat or wide-but-bell priors are fine — the issue is
distributional concentration, not range.

## Experiment 2: extra cond dim (Makkunda kernel, narrow ls, blks=4, 3-seed mean)

| extra cond | seed 19 | seed 42 | seed 7 | FM mean | gMLP mean |
|---|---|---|---|---|---|
| none (2D) | 0.18 | 0.17 | 0.26 | 0.20 | 0.010 |
| +a Uniform(0.8, 1.2) | 0.086 | 0.083 | 0.142 | **0.10** | 0.011 |
| +nu Uniform(1.0, 2.0) | 0.124 | 0.615 | 0.404 | 0.38 | 0.010 |
| +a+nu (4D) | 0.332 | 0.233 | 0.233 | 0.27 | 0.011 |

Extra single experiments:
- +a Uniform(0.5, 2.0): FM=0.43 — wide a hurts
- +nu Uniform(0.8, 1.2): NaN — violates Gneiting PSD (ν ≥ d/2 = 1 in 2D)
- 4D blks=8: FM=0.07 — capacity rescues

**Takeaways**:
- Tight `a` cond helps FM (0.10 < 0.20). Wide `a` hurts (0.43).
- `nu` is unstable across seeds even when tight; sometimes great (0.12),
  sometimes bad (0.62). Not robust.
- 4D fails at blks=4 but works at blks=8 — capacity-bound.
- gMLP is rock-solid (0.010 ± 0.002) across everything.

## Experiment 3: kernel form — Makkunda (Gaussian spatial) vs ours (Matérn-1/2)

3-seed mean, narrow ls:

| kernel | extra cond | gMLP MSE | FM MSE |
|---|---|---|---|
| Makkunda (Gaussian spatial) | none (2D) | 0.010 | 0.20 |
| Makkunda | +a tight (3D) | 0.011 | 0.10 |
| ours (Matérn-1/2) | none (2D, ls_t fixed) | **0.005** | **0.12** |
| ours | +lst (3D) | 0.007 | **0.10** |

**Takeaway**: our kernel improves both decoders.
- gMLP: 2× better (0.005 vs 0.010)
- FM: ~40% better at 2D (0.12 vs 0.20); equal at 3D (~0.10)

Exponential spatial decay leaves more local variation than Gaussian/RBF,
which gives both decoders a richer training distribution.

## Summary

Three independent levers for getting FM to train cleanly:
1. **Prior shape**: avoid right-skewed priors that pile mass at boundaries
   (Beta(4,1)+LogScale was the original killer).
2. **Cond design**: tight prior on a single extra dim (e.g., `a`) helps;
   wide priors or 4D-at-blks=4 hurt.
3. **Kernel form**: Matérn-1/2 spatial > Gaussian/RBF spatial for both
   gMLP and FM at this benchmark.

The fix-fm-yogi-narrowpriors branch achieves (1) via the 2D
parameterisation. Combining (1+2+3) gives the cleanest result.
