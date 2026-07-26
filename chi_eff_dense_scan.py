"""
chi_eff_dense_scan.py — χ_eff(T) Dense Scan @ Tc Window
驗證雙峰結構：T=2.25-2.37, ΔT=0.01, N=500, p_nuc=0.05
"""
import numpy as np
import json, time
from numpy.linalg import svd

# ─── params ───
L = 32
N_total = 500
p_nuc = 0.05
T_list = np.arange(2.25, 2.375, 0.01)  # 2.25, 2.26, ..., 2.37

SEED = 42
np.random.seed(SEED)

def generate_ising_thermal(L, T, N):
    """Generate N independent Ising configurations at temperature T using Metropolis MC."""
    configs = np.zeros((N, L*L), dtype=np.float64)
    # Initial: random
    spins = np.random.choice([-1, 1], size=(L, L))
    
    # Equilibrate
    for _ in range(1000):
        i, j = np.random.randint(0, L, 2)
        dE = 2 * spins[i,j] * (
            spins[(i+1)%L, j] + spins[(i-1)%L, j] + 
            spins[i, (j+1)%L] + spins[i, (j-1)%L]
        )
        if dE <= 0 or np.random.random() < np.exp(-dE / T):
            spins[i,j] *= -1
    
    # Sample
    for n in range(N):
        for _ in range(L*L):
            i, j = np.random.randint(0, L, 2)
            dE = 2 * spins[i,j] * (
                spins[(i+1)%L, j] + spins[(i-1)%L, j] + 
                spins[i, (j+1)%L] + spins[i, (j-1)%L]
            )
            if dE <= 0 or np.random.random() < np.exp(-dE / T):
                spins[i,j] *= -1
        configs[n] = spins.flatten()
    return configs

def embed_nuclei(configs, p, L):
    """Embed p*N vacuum-like nuclei (random, no spatial correlation) into each config."""
    N, D = configs.shape
    n_vac = int(N * p)
    configs_nuc = configs.copy()
    # Replace n_vac rows with vacuum-like samples (i.i.d. Gaussian ~ vacuum)
    vac_nuclei = np.random.randn(n_vac, D)
    configs_nuc[-n_vac:] = vac_nuclei
    return configs_nuc

def compute_ohat(X):
    """Compute Ô-HAT metrics: TopSV, helicity, theta1 relative to reference."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = svd(X_centered, full_matrices=False)
    S2 = S**2
    total = S2.sum()
    TopSV = S2[0] / total
    
    # Helicity: signed skew of singular values
    k = min(len(S), 20)
    y = np.log(S[:k] + 1e-16)
    x = np.arange(k)
    slope = np.polyfit(x, y, 1)[0]
    helicity = -slope  # positive = more structure
    
    # EffRank
    S2_norm = S2 / total
    entropy = -np.sum(S2_norm * np.log(S2_norm + 1e-16))
    EffRank = int(np.exp(entropy))
    
    return TopSV, helicity, EffRank, Vt[0]

# ─── Main ───
results = []
t_start = time.time()

for idx, T in enumerate(T_list):
    t0 = time.time()
    print(f"[{idx+1}/{len(T_list)}] T={T:.2f} ...", end=" ", flush=True)
    
    # Generate pure matter
    matter = generate_ising_thermal(L, T, N_total)
    
    # Embed nuclei
    matter_nuc = embed_nuclei(matter, p_nuc, L)
    
    # Ô-HAT on pure matter
    TopSV_0, helicity_0, EffRank_0, v0 = compute_ohat(matter)
    
    # Ô-HAT on matter+nuclei
    TopSV_nuc, helicity_nuc, EffRank_nuc, v_nuc = compute_ohat(matter_nuc)
    
    # θ₁: angle between pure matter V1 and matter+nuclei V1
    # Both are from same configs except nuclei rows → compare row-subspace
    # Actually: compute θ₁ between the two configuration matrices' V1
    dot = np.abs(np.dot(v0, v_nuc))
    dot = np.clip(dot, 0, 1)
    theta1_nuc = np.degrees(np.arccos(dot))
    delta_theta = 90 - theta1_nuc if theta1_nuc < 90 else theta1_nuc - 90
    # Actually keep raw theta1, compute chi_eff from 90-theta1
    # But v0 vs itself = 0°, so delta from pure matter is just theta1_nuc
    
    chi_eff = theta1_nuc / p_nuc  # degrees per unit p
    
    dt = time.time() - t0
    print(f"θ₁={theta1_nuc:.1f}°  χ_eff={chi_eff:.1f}  TopSV_nuc={TopSV_nuc:.3f}  [{dt:.0f}s]")
    
    results.append({
        "T": float(T),
        "TopSV_0": float(TopSV_0),
        "helicity_0": float(helicity_0),
        "theta1_nuc": float(theta1_nuc),
        "delta_theta": float(theta1_nuc),  # from pure matter
        "chi_eff": float(chi_eff),
        "TopSV_nuc": float(TopSV_nuc),
        "helicity_nuc": float(helicity_nuc),
        "time_s": dt
    })

total_s = time.time() - t_start

output = {
    "experiment": "χ_eff(T) Dense Scan — Double-Peak Verification",
    "params": {
        "L": L,
        "N": N_total,
        "p_nuc": p_nuc,
        "T_range": [float(T_list[0]), float(T_list[-1])],
        "T_step": 0.01,
        "n_points": len(T_list)
    },
    "results": results,
    "timing": {"total_s": total_s}
}

outpath = "chi_eff_dense_Tc_window.json"
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Done in {total_s:.0f}s → {outpath}")
print(f"   T range: {T_list[0]:.2f} → {T_list[-1]:.2f} ({len(T_list)} pts)")
print(f"   p_nuc={p_nuc}, N={N_total}")
