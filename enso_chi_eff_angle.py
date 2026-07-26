#!/usr/bin/env python3
"""
enso_chi_eff_angle.py — Ô-HAT χ_eff (Bridge Angle) Measurement for ENSO
=======================================================================

Adapts the Ising χ_eff framework to ENSO time series:
  1. Convert ONI 1D → configuration matrix via delay embedding (Takens)
  2. Embed vacuum nuclei (white noise replacing p*N configurations)
  3. Compute SVD → θ₁ (angle between pure V1 and perturbed V1)
  4. χ_eff = θ₁ / p_nuc  (geometric susceptibility)

Tests paper Conjecture 2 (universality): does χ_eff peak in a specific ENSO phase,
analogous to post-Tc peak in Ising?

Two experiments:
  A. Phase-level χ_eff: compute χ_eff separately for El Niño / Neutral / La Niña
  B. P-sweep: measure θ₁(p) curve across p=[0.01, 0.50] for each phase
"""

import numpy as np
import json
import time
from pathlib import Path
from numpy.linalg import svd

# ─── Config ───
WINDOW_LENGTH = 12      # months per configuration vector (1 year)
STRIDE = 3              # months stride (finer than 6 for more configurations)
P_NUC = 0.05            # vacuum nucleus fraction
N_BOOTSTRAP = 30        # bootstrap resamples for error bars
P_SWEEP = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SEED = 42

BASE = Path("/app/working/workspaces/tygtDc/projects/enso")
DATA = BASE / "data" / "oni.csv"
OUTPUT = BASE / "output" / "enso_chi_eff.json"

# ─── 1. Load ONI ───
def load_oni(path):
    years, months, values = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if year < 1950 or year > 2026:
                continue
            for m, val_str in enumerate(parts[1:13], 1):
                try:
                    val = float(val_str)
                    if val > -99:
                        years.append(year)
                        months.append(m)
                        values.append(val)
                except ValueError:
                    continue
    return np.array(values)

# ─── 2. Delay embedding → configuration matrix ───
def delay_embed(series, window, stride):
    """Convert 1D series to N×window configuration matrix via sliding windows."""
    n = (len(series) - window) // stride + 1
    configs = np.zeros((n, window))
    for i in range(n):
        start = i * stride
        configs[i] = series[start:start + window]
    return configs

# ─── 3. Phase classification ───
def classify_phase(oni_windows):
    """Classify each window by its mean ONI value."""
    means = oni_windows.mean(axis=1)
    el_nino = means >= 0.5
    la_nina = means <= -0.5
    neutral = ~(el_nino | la_nina)
    return el_nino, neutral, la_nina

# ─── 4. Vacuum nucleus embedding ───
def embed_nuclei(configs, p, seed=None):
    """Replace p*N rows with vacuum (i.i.d. Gaussian) nuclei."""
    if seed is not None:
        np.random.seed(seed)
    N, D = configs.shape
    n_vac = max(1, int(N * p))
    configs_nuc = configs.copy()
    vac_nuclei = np.random.randn(n_vac, D)
    # Randomly select rows to replace
    idx = np.random.choice(N, n_vac, replace=False)
    configs_nuc[idx] = vac_nuclei
    return configs_nuc

# ─── 5. SVD → θ₁ ───
def compute_theta1(X_pure, X_nuc):
    """Compute θ₁ = angle between V1 of pure subspace and V1 of perturbed."""
    # Center
    Xp = X_pure - X_pure.mean(axis=0)
    Xn = X_nuc - X_nuc.mean(axis=0)
    
    _, _, Vt_pure = svd(Xp, full_matrices=False)
    _, _, Vt_nuc = svd(Xn, full_matrices=False)
    
    v1_pure = Vt_pure[0]
    v1_nuc = Vt_nuc[0]
    
    dot = np.abs(np.dot(v1_pure, v1_nuc))
    dot = np.clip(dot, 0, 1)
    theta1 = np.degrees(np.arccos(dot))
    return theta1

# ─── 6. Chi_eff at fixed p ───
def chi_eff_single(configs, p, n_bootstrap=N_BOOTSTRAP):
    """Compute χ_eff = θ₁/p with bootstrap error."""
    N = len(configs)
    values = []
    
    for b in range(n_bootstrap):
        seed_b = SEED * 10000 + b * 100 + 1
        np.random.seed(seed_b)
        # Bootstrap resample
        idx = np.random.choice(N, N, replace=True)
        configs_b = configs[idx]
        
        # Embed nuclei
        seed_nuc = SEED * 10000 + b * 100 + 2
        configs_nuc = embed_nuclei(configs_b, p, seed=seed_nuc)
        
        theta1 = compute_theta1(configs_b, configs_nuc)
        values.append(theta1 / p)
    
    values = np.array(values)
    return {
        "chi_eff_mean": float(values.mean()),
        "chi_eff_std": float(values.std()),
        "chi_eff_ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "theta1_mean": float(values.mean() * p),
        "theta1_std": float(values.std() * p),
    }

# ─── 7. P-sweep ───
def chi_eff_psweep(configs, p_list, n_bootstrap=N_BOOTSTRAP):
    """Compute χ_eff across a range of p values."""
    results = []
    for p in p_list:
        r = chi_eff_single(configs, p, n_bootstrap=n_bootstrap)
        r["p"] = p
        results.append(r)
        print(f"    p={p:.2f}  χ_eff={r['chi_eff_mean']:.1f} ± {r['chi_eff_std']:.1f}  θ₁={r['theta1_mean']:.2f}°")
    return results

# ─── Main ───
def main():
    t_start = time.time()
    np.random.seed(SEED)
    
    print("=== ENSO χ_eff (Bridge Angle) Measurement ===")
    print(f"WINDOW={WINDOW_LENGTH}mo, STRIDE={STRIDE}mo, P_NUC={P_NUC}, BOOTSTRAP={N_BOOTSTRAP}x\n")
    
    # Load
    oni = load_oni(DATA)
    print(f"[1] Loaded ONI: {len(oni)} months ({oni.min():.2f} to {oni.max():.2f})")
    
    # Delay embed
    configs = delay_embed(oni, WINDOW_LENGTH, STRIDE)
    print(f"[2] Delay embedding: {configs.shape[0]} configurations × {configs.shape[1]} dims")
    
    # Classify
    el_nino, neutral, la_nina = classify_phase(configs)
    print(f"[3] Phase split: El Niño={el_nino.sum()}  Neutral={neutral.sum()}  La Niña={la_nina.sum()}")
    
    phases = {
        "El_Nino": configs[el_nino],
        "Neutral": configs[neutral],
        "La_Nina": configs[la_nina],
        "All": configs  # combined for reference
    }
    
    output = {
        "experiment": "ENSO χ_eff (Bridge Angle) — Cross-Domain Universality Test",
        "params": {
            "window_length_months": WINDOW_LENGTH,
            "stride_months": STRIDE,
            "p_nuc": P_NUC,
            "n_bootstrap": N_BOOTSTRAP,
            "oni_range": [float(oni.min()), float(oni.max())],
            "oni_length": len(oni),
        },
        "phase_counts": {
            "El_Nino": int(el_nino.sum()),
            "Neutral": int(neutral.sum()),
            "La_Nina": int(la_nina.sum()),
            "Total": int(el_nino.sum() + neutral.sum() + la_nina.sum())
        },
    }
    
    # ─── Experiment A: Phase-level χ_eff ───
    print("\n─── Experiment A: Phase-level χ_eff (p=0.05) ───")
    phase_chi = {}
    for name, mat in phases.items():
        if len(mat) < 20:
            print(f"  {name}: SKIP (n={len(mat)} < 20)")
            continue
        print(f"\n  {name} (n={len(mat)}):")
        r = chi_eff_single(mat, P_NUC)
        phase_chi[name] = r
        print(f"    χ_eff = {r['chi_eff_mean']:.1f} ± {r['chi_eff_std']:.1f}  [95% CI: {r['chi_eff_ci95']}]")
    output["phase_level"] = phase_chi
    
    # ─── Experiment B: P-sweep per phase ───
    print("\n─── Experiment B: P-sweep ───")
    psweep = {}
    for name, mat in phases.items():
        if len(mat) < 20:
            continue
        print(f"\n  {name} p-sweep (n={len(mat)}):")
        psweep[name] = chi_eff_psweep(mat, P_SWEEP)
    output["p_sweep"] = psweep
    
    # ─── Save ───
    output["timing"] = {"total_s": time.time() - t_start}
    
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved → {OUTPUT}")
    print(f"   Total time: {output['timing']['total_s']:.0f}s")

if __name__ == "__main__":
    main()
