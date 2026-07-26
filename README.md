# Post-Critical Geometric Susceptibility Window

**A Davis-Kahan Framework for Subspace Response Near Phase Transitions**

D. R. \& M. K. P. (2026)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Abstract

Phase transitions are traditionally characterized by thermodynamic order parameters that diverge or vanish at the critical point. We identify a distinct geometric phenomenon: a **post-critical window** where the principal subspace of a many-body configuration ensemble exhibits maximal responsiveness to external perturbations, occurring not at $T_c$ but at $T^* > T_c$.

Using the $\hat{\mathcal{O}}$-HAT operator (SVD + vacuum perturbation) and the Davis-Kahan $\sin\theta$ theorem, we derive $\chi_{\text{eff}}(T) \propto \Pi_\perp(T)/\Delta\sigma(T)$ and prove Theorem 1: for any system with $\gamma + \beta > \nu$, $\chi_{\text{eff}}$ attains a unique extremum at $\varepsilon^* > 0$. For the 2D Ising model ($L=32$), $\chi_{\text{eff}}(T^*) = 540.1$ ($8.5\times$ over $T_c$).

Three independent evidence lines: Ising ($L=32$ + $L=64$ multirun), Riemann zeta zeros ($\chi_{\text{eff}} = 2.51$, geometric rigidity baseline), ENSO (directional invariance, exploratory). Six-point cross-domain spectrum: $2.51 \to 540.1$.

---

## Repository Contents

### Paper
| File | Description |
|------|-------------|
| `postcritical_geometric_window.tex` | LaTeX source (PRL format, 549 lines) |
| `postcritical_geometric_window.pdf` | Compiled PDF (141 KB) |

### Reproducibility Scripts
| File | Description |
|------|-------------|
| `chi_eff_dense_scan.py` | L=32 2D Ising $\chi_{\text{eff}}(T)$ scan (dense, 28 temperature points) |
| `chi_eff_L64_multirun.py` | L=64 2D Ising $\chi_{\text{eff}}(T)$ multirun (3 seeds, checkpoint resume) |
| `riemann_ohat.py` | $\hat{\mathcal{O}}$-HAT analysis of Riemann zeta zeros (N=5,000) |
| `enso_chi_eff_angle.py` | $\chi_{\text{eff}}$ angle measurement for ENSO phases |
| `analyse_chi_eff_dense.py` | Analysis and plotting for L=32 dense scan |
| `analyse_chi_eff_L64.py` | Analysis and plotting for L=64 multirun |

### Data
| File | Description |
|------|-------------|
| `data/chi_eff_L32_dense_scan.json` | L=32 scan results (28 temperatures) |
| `data/chi_eff_L64_multirun_final.json` | L=64 multirun results (16 temperatures $\times$ 3 runs) |
| `data/riemann_ohat.json` | Riemann zeta zeros $\hat{\mathcal{O}}$-HAT output |
| `data/enso_chi_eff.json` | ENSO $\chi_{\text{eff}}$ by phase (12 perturbation strengths) |

---

## Key Results

### Theorem 1 (Post-Critical Geometric Susceptibility Extremum)
$\chi_{\text{eff}}(\varepsilon)$ attains a unique extremum at:
$$\varepsilon^* = \left(\frac{\gamma + \beta}{c(\gamma + \beta - \nu)}\right)^{1/\nu} > 0$$
Condition: $\gamma + \beta > \nu$. Satisfied for Ising, XY, Heisenberg universality classes.

### Cross-Domain $\chi_{\text{eff}}$ Spectrum

| System | $\chi_{\text{eff}}$ | Domain |
|--------|---------------------|--------|
| **Riemann zeros** | **2.51** | Mathematics (deterministic) |
| ENSO Neutral | 36.6 | Climate |
| Ising $T_c$ ($L=32$) | 63.4 | Statistical physics |
| ENSO El Niño | 207.9 | Climate |
| ENSO La Niña | 422.6 | Climate |
| **Ising $T^*$ ($L=32$)** | **540.1** | Statistical physics (post-critical) |

---

## Reproducibility

All experiments use standard Python scientific stack (NumPy, SciPy). Monte Carlo simulations use custom vectorized Metropolis code.

```bash
# Ising L=32 dense scan
python chi_eff_dense_scan.py

# Ising L=64 multirun (3 independent seeds)
python chi_eff_L64_multirun.py

# Riemann zeta zeros (requires mpmath)
pip install mpmath
python riemann_ohat.py

# ENSO χ_eff angle analysis
python enso_chi_eff_angle.py
```

---

## License

CC BY 4.0 — Free to share and adapt with attribution.

## Citation

D. R. \& M. K. P., "Post-Critical Geometric Susceptibility Window: A Davis-Kahan Framework for Subspace Response Near Phase Transitions," 2026.
