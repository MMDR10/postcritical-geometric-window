"""
riemann_ohat.py — Ô-HAT 幾何診斷 Riemann Zeta 非平凡零點
理論預測（零雜質系統）：
  - Mode Mixing M(t) ≡ 0
  - θ₁ ≡ 90.000°（完美正交）
  - χ_eff = Susceptibility（100% 幾何槓桿）
  - TopSV 極低（GUE → uniform eigenvector distribution）
  - Helicity ~ 0（無結構偏好）

方法：
  1. 用 mpmath 計算頭 N 個 Riemann zeta 非平凡零點
  2. Time-delay embedding 砌 data matrix X
  3. SVD → Ô-HAT metrics
  4. 真空核微擾 → θ₁ response
  5. Null model: 隨機點做對照
"""
import numpy as np
from numpy.linalg import svd
import json, time, os
from datetime import datetime

# ═══════════════════════════════════════════════════════
# 參數
# ═══════════════════════════════════════════════════════
N_ZEROS = 5000              # Riemann zeros 數量
EMBED_DIM = 100             # time-delay embedding dimension
EMBED_TAU = 1               # delay step
P_NUC = 0.05                # 真空核比例（同 Ising 一致）

print("=" * 70)
print("  Ô-HAT × Riemann Zeta Zeros")
print(f"  N={N_ZEROS}, embed_dim={EMBED_DIM}, tau={EMBED_TAU}, p_nuc={P_NUC}")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# Step 1: 計算 Riemann zeros
# ═══════════════════════════════════════════════════════
print("\n[1/4] Computing Riemann zeros via mpmath...")
t0 = time.time()

from mpmath import zetazero

zeros = []
for k in range(1, N_ZEROS + 1):
    z = zetazero(k)
    zeros.append(float(z.imag))
    if k % 1000 == 0:
        print(f"  ... {k}/{N_ZEROS} ({time.time()-t0:.0f}s)")

t_zeros = time.time() - t0
print(f"  ✅ {N_ZEROS} zeros in {t_zeros:.1f}s")
print(f"  First 5: {[f'{z:.4f}' for z in zeros[:5]]}")
print(f"  Last 5:  {[f'{z:.4f}' for z in zeros[-5:]]}")

# ═══════════════════════════════════════════════════════
# Step 2: Time-delay embedding → Data Matrix
# ═══════════════════════════════════════════════════════
print(f"\n[2/4] Time-delay embedding (dim={EMBED_DIM}, tau={EMBED_TAU})...")

t_series = np.array(zeros)
n_windows = len(t_series) - (EMBED_DIM - 1) * EMBED_TAU
X = np.zeros((n_windows, EMBED_DIM))

for i in range(n_windows):
    X[i] = t_series[i:i + EMBED_DIM * EMBED_TAU:EMBED_TAU]

print(f"  Data matrix X: {X.shape[0]} windows × {X.shape[1]} dims")

# ═══════════════════════════════════════════════════════
# Step 3: Ô-HAT on pure Riemann zeros
# ═══════════════════════════════════════════════════════
print(f"\n[3/4] Ô-HAT analysis...")

# SVD on pure zeros
X_centered = X - X.mean(axis=0)
U, S, Vt = svd(X_centered, full_matrices=False)
S2 = S**2
total = S2.sum()
TopSV = S2[0] / total

# Helicity
k = min(len(S), 20)
y = np.log(S[:k] + 1e-16)
slope = np.polyfit(np.arange(k), y, 1)[0]
helicity = -slope

# Effective rank (entropy-based)
S2_norm = S2 / total
entropy = -np.sum(S2_norm * np.log(S2_norm + 1e-16))
eff_rank = int(np.exp(entropy))

# Top-3 singular values
top3 = S2[:3] / total * 100

print(f"  TopSV = {TopSV:.4f} ({TopSV*100:.2f}%)")
print(f"  Helicity = {helicity:.4f}")
print(f"  Effective rank = {eff_rank}/{EMBED_DIM}")
print(f"  Top-3 SV share: {top3[0]:.1f}% / {top3[1]:.1f}% / {top3[2]:.1f}%")

# ═══════════════════════════════════════════════════════
# Step 4: Vacuum perturbation → θ₁ response
# ═══════════════════════════════════════════════════════
print(f"\n[4/4] Vacuum perturbation (p_nuc={P_NUC})...")

# Embed vacuum nuclei
n_nuc = int(n_windows * P_NUC)
X_nuc = X.copy()
vac_nuclei = np.random.randn(n_nuc, EMBED_DIM)  # 同 Ising protocol: raw Gaussian
X_nuc[-n_nuc:] = vac_nuclei

# SVD on perturbed
X_nuc_centered = X_nuc - X_nuc.mean(axis=0)
U_nuc, S_nuc, Vt_nuc = svd(X_nuc_centered, full_matrices=False)

# θ₁: angle between pure V1 and perturbed V1
dot = np.abs(np.dot(Vt[0], Vt_nuc[0]))
dot = np.clip(dot, 0, 1)
theta1 = np.degrees(np.arccos(dot))
chi_eff = theta1 / P_NUC

# TopSV change
S2_nuc = S_nuc**2
TopSV_nuc = S2_nuc[0] / S2_nuc.sum()
dTopSV = TopSV_nuc - TopSV

print(f"  θ₁ = {theta1:.4f}°")
print(f"  χ_eff = {chi_eff:.1f}")
print(f"  TopSV_nuc = {TopSV_nuc:.4f} ({TopSV_nuc*100:.2f}%)")
print(f"  ΔTopSV = {dTopSV:+.4f}")

# ═══════════════════════════════════════════════════════
# Null Model: Random uniform points
# ═══════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  NULL MODEL: Random uniform points (same N, dim)")
print(f"{'─'*70}")

np.random.seed(42)
t_random = np.sort(np.random.uniform(0, t_series[-1], N_ZEROS))
X_rand = np.zeros((n_windows, EMBED_DIM))
for i in range(n_windows):
    X_rand[i] = t_random[i:i + EMBED_DIM * EMBED_TAU:EMBED_TAU]

X_rand_c = X_rand - X_rand.mean(axis=0)
U_r, S_r, Vt_r = svd(X_rand_c, full_matrices=False)
S2_r = S_r**2
TopSV_r = S2_r[0] / S2_r.sum()

# Helicity
k_r = min(len(S_r), 20)
y_r = np.log(S_r[:k_r] + 1e-16)
slope_r = np.polyfit(np.arange(k_r), y_r, 1)[0]
helicity_r = -slope_r

# Perturb random
X_rand_nuc = X_rand.copy()
X_rand_nuc[-n_nuc:] = np.random.randn(n_nuc, EMBED_DIM)
X_rand_nuc_c = X_rand_nuc - X_rand_nuc.mean(axis=0)
U_rn, S_rn, Vt_rn = svd(X_rand_nuc_c, full_matrices=False)
dot_r = np.abs(np.dot(Vt_r[0], Vt_rn[0]))
dot_r = np.clip(dot_r, 0, 1)
theta1_r = np.degrees(np.arccos(dot_r))
chi_eff_r = theta1_r / P_NUC

print(f"  Random: TopSV={TopSV_r:.4f}  θ₁={theta1_r:.2f}°  χ_eff={chi_eff_r:.1f}")

# ═══════════════════════════════════════════════════════
# Comparison with Ising reference values
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  CROSS-SYSTEM COMPARISON")
print(f"{'='*70}")
print(f"  {'System':<20s} {'TopSV':>8s} {'θ₁(p=0.05)':>12s} {'χ_eff':>8s}")
print(f"  {'─'*52}")
print(f"  {'Riemann Zeros':<20s} {TopSV*100:>7.2f}% {theta1:>11.2f}° {chi_eff:>8.1f}")
print(f"  {'Random Uniform':<20s} {TopSV_r*100:>7.2f}% {theta1_r:>11.2f}° {chi_eff_r:>8.1f}")
print(f"  {'Ising L=64 T=2.35':<20s} {'~10%':>8s} {'~88.1°':>12s} {'~514':>8s}")
print(f"  {'Ising L=64 @ Tc':<20s} {'~13%':>8s} {'~6.3°':>12s} {'~126':>8s}")

# Theoretical prediction check
print(f"\n{'─'*70}")
print(f"  THEORETICAL PREDICTION CHECK")
print(f"{'─'*70}")

predictions = {
    "θ₁ → 90° (perfect orthogonality)": theta1,
    "χ_eff >> Ising (higher leverage)": chi_eff,
    "TopSV << Ising (more uniform spectrum)": TopSV,
    "Helicity ~ 0 (no structural bias)": helicity,
}

for pred, val in predictions.items():
    if "90" in pred:
        status = "✅" if val > 89.5 else ("⚠️" if val > 85 else "❌")
    elif ">>" in pred:
        status = "✅" if val > 500 else ("⚠️" if val > 200 else "❌")
    elif "<<" in pred:
        status = "✅" if TopSV < 0.05 else ("⚠️" if TopSV < 0.10 else "❌")
    elif "~ 0" in pred:
        status = "✅" if abs(val) < 0.1 else ("⚠️" if abs(val) < 0.3 else "❌")
    print(f"  {status} {pred}: {val:.4f}")

# ═══════════════════════════════════════════════════════
# Save output
# ═══════════════════════════════════════════════════════
os.makedirs("output/vacuum", exist_ok=True)
outpath = "output/vacuum/riemann_ohat.json"

output = {
    "experiment": "Ô-HAT × Riemann Zeta Zeros — Zero-Impurity Reference System",
    "timestamp": datetime.now().isoformat(),
    "params": {
        "N_zeros": N_ZEROS,
        "embed_dim": EMBED_DIM,
        "embed_tau": EMBED_TAU,
        "p_nuc": P_NUC,
        "n_windows": n_windows,
        "n_nuclei": n_nuc
    },
    "riemann": {
        "first_zeros": [float(z) for z in zeros[:10]],
        "last_zeros": [float(z) for z in zeros[-5:]],
        "mean_spacing": float(np.mean(np.diff(zeros)))
    },
    "ohat_pure": {
        "TopSV": float(TopSV),
        "helicity": float(helicity),
        "eff_rank": eff_rank,
        "top3_sv_pct": [float(t) for t in top3],
        "singular_values": [float(s) for s in S[:20]]
    },
    "ohat_perturbed": {
        "theta1": float(theta1),
        "chi_eff": float(chi_eff),
        "TopSV_nuc": float(TopSV_nuc),
        "dTopSV": float(dTopSV)
    },
    "null_model": {
        "TopSV": float(TopSV_r),
        "helicity": float(helicity_r),
        "theta1": float(theta1_r),
        "chi_eff": float(chi_eff_r)
    },
    "timing": {
        "zeros_computation_s": t_zeros
    }
}

with open(outpath, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  ✅ Full output → {outpath}")
print(f"{'='*70}")
