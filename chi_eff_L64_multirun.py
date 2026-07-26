"""
chi_eff_L64_multirun.py — L=64 χ_eff(T) Multi-Run Statistical Scan
目的：5 independent MC runs, average χ_eff over runs → 消滅 single-run noise
     驗證 peak 是否 physical 定 noise artifact
用法：python3 chi_eff_L64_multirun.py
中途 save：每完成一個 run 就 save checkpoint，斷咗可以 resume
"""

import numpy as np
import json, time, os, sys
from numpy.linalg import svd
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# 可調參數
# ═══════════════════════════════════════════════════════════════
L = 64
N_total = 400           # sample 數量（輕量版，足夠做 statistics）
p_nuc = 0.05
eq_sweeps = 2000        # equilibration sweeps
sample_sweeps = 3       # 每個 sample 之間嘅 decorrelation sweeps
T_start, T_end = 2.30, 2.375
T_step = 0.005
N_RUNS = 3              # independent runs 數量（3 runs 夠計 std）
SEED_BASE = 4200        # seed = SEED_BASE + run_index

# ═══════════════════════════════════════════════════════════════
T_list = np.arange(T_start, T_end + T_step/2, T_step)
T_list = [round(float(t), 5) for t in T_list]  # avoid float drift
N_POINTS = len(T_list)

print("=" * 72)
print(f"  χ_eff(T) L={L} MULTI-RUN STATISTICAL SCAN")
print(f"  N={N_total}, p_nuc={p_nuc}, {sample_sweeps} sweeps/sample")
print(f"  T: [{T_start}, {T_end}] ΔT={T_step} ({N_POINTS} pts)")
print(f"  Runs: {N_RUNS} independent, SEED_BASE={SEED_BASE}")
print(f"  est. time/run: ~{N_POINTS * N_total/300 * sample_sweeps * 0.13:.0f}s")
print("=" * 72)

def generate_ising_thermal(L, T, N, sweeps_between):
    """Generate N decorrelated Ising configs at temperature T."""
    configs = np.zeros((N, L*L), dtype=np.float64)
    spins = np.random.choice([-1, 1], size=(L, L))
    
    for _ in range(eq_sweeps):
        i, j = np.random.randint(0, L, 2)
        dE = 2 * spins[i,j] * (
            spins[(i+1)%L, j] + spins[(i-1)%L, j] + 
            spins[i, (j+1)%L] + spins[i, (j-1)%L]
        )
        if dE <= 0 or np.random.random() < np.exp(-dE / T):
            spins[i,j] *= -1
    
    for n in range(N):
        for _ in range(L*L * sweeps_between):
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
    N, D = configs.shape
    n_vac = int(N * p)
    configs_nuc = configs.copy()
    vac_nuclei = np.random.randn(n_vac, D)
    configs_nuc[-n_vac:] = vac_nuclei
    return configs_nuc

def compute_ohat(X):
    X_centered = X - X.mean(axis=0)
    U, S, Vt = svd(X_centered, full_matrices=False)
    S2 = S**2
    total = S2.sum()
    TopSV = S2[0] / total
    k = min(len(S), 20)
    y = np.log(S[:k] + 1e-16)
    slope = np.polyfit(np.arange(k), y, 1)[0]
    helicity = -slope
    return TopSV, helicity, Vt[0]

# ═══════════════════════════════════════════════════════════════
# Checkpoint: load existing progress
# ═══════════════════════════════════════════════════════════════
CHECKPOINT = "output/vacuum/chi_eff_L64_multirun_checkpoint.json"
all_runs = []
completed_runs = set()

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    all_runs = ckpt.get("runs", [])
    completed_runs = {r["run_index"] for r in all_runs}
    print(f"\n📂 Resuming from checkpoint: {len(all_runs)}/{N_RUNS} runs done")
    print(f"   Completed runs: {sorted(completed_runs)}")

# ═══════════════════════════════════════════════════════════════
# Main loop over runs
# ═══════════════════════════════════════════════════════════════
for run_idx in range(N_RUNS):
    if run_idx in completed_runs:
        continue
    
    seed = SEED_BASE + run_idx
    np.random.seed(seed)
    
    print(f"\n{'─'*72}")
    print(f"  RUN {run_idx+1}/{N_RUNS}  (seed={seed})  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'─'*72}")
    print(f"  {'T':>8s}  {'θ₁':>7s}  {'χ_eff':>8s}  {'TopSV₀':>8s}  {'TopSVₙ':>8s}  {'t':>5s}")
    
    run_results = []
    run_start = time.time()
    
    for idx, T in enumerate(T_list):
        t0 = time.time()
        
        matter = generate_ising_thermal(L, T, N_total, sample_sweeps)
        matter_nuc = embed_nuclei(matter, p_nuc, L)
        
        TopSV_0, helicity_0, v0 = compute_ohat(matter)
        TopSV_nuc, helicity_nuc, v_nuc = compute_ohat(matter_nuc)
        
        dot = np.abs(np.dot(v0, v_nuc))
        dot = np.clip(dot, 0, 1)
        theta1 = np.degrees(np.arccos(dot))
        chi_eff = theta1 / p_nuc
        
        dt = time.time() - t0
        print(f"  {T:8.3f}  {theta1:7.1f}  {chi_eff:8.1f}  {TopSV_0:8.3f}  {TopSV_nuc:8.3f}  {dt:4.0f}s")
        
        run_results.append({
            "T": T,
            "theta1": float(theta1),
            "chi_eff": float(chi_eff),
            "TopSV_0": float(TopSV_0),
            "TopSV_nuc": float(TopSV_nuc),
            "helicity_0": float(helicity_0),
            "helicity_nuc": float(helicity_nuc),
            "time_s": dt
        })
    
    run_time = time.time() - run_start
    all_runs.append({
        "run_index": run_idx,
        "seed": seed,
        "results": run_results,
        "run_time_s": run_time
    })
    completed_runs.add(run_idx)
    
    # Save checkpoint after each run
    os.makedirs("output/vacuum", exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({
            "params": {
                "L": L, "N": N_total, "p_nuc": p_nuc,
                "eq_sweeps": eq_sweeps,
                "sample_sweeps": sample_sweeps,
                "T_start": T_start, "T_end": T_end, "T_step": T_step,
                "N_RUNS": N_RUNS, "SEED_BASE": SEED_BASE
            },
            "runs": all_runs,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"  ✅ Run {run_idx+1} done in {run_time:.0f}s → checkpoint saved")

# ═══════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print(f"  STATISTICAL ANALYSIS ({len(all_runs)} runs)")
print(f"{'='*72}")

# Aggregate: for each T, collect chi_eff from all runs
T_vals = T_list
agg = {T: [] for T in T_vals}
for run in all_runs:
    for pt in run["results"]:
        agg[pt["T"]].append(pt["chi_eff"])

print(f"\n  {'T':>8s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}  {'CV%':>6s}")
print(f"  {'─'*56}")

stats = []
for T in T_vals:
    vals = np.array(agg[T])
    mean = vals.mean()
    std = vals.std(ddof=1)
    cv = (std / mean * 100) if mean > 0 else 0
    print(f"  {T:8.3f}  {mean:8.1f}  {std:8.1f}  {vals.min():8.1f}  {vals.max():8.1f}  {cv:5.1f}%")
    stats.append({"T": T, "mean": float(mean), "std": float(std), 
                  "min": float(vals.min()), "max": float(vals.max()),
                  "cv_pct": float(cv), "values": [float(v) for v in vals]})

# Find peak in mean
means = np.array([s["mean"] for s in stats])
idx_peak = np.argmax(means)
print(f"\n  🏔  Mean χ_eff peak: {means[idx_peak]:.1f} @ T={T_vals[idx_peak]:.3f}")
print(f"      (σ = {stats[idx_peak]['std']:.1f}, CV = {stats[idx_peak]['cv_pct']:.1f}%)")

# Is the peak statistically significant?
# Compare peak value to baseline
baseline_mask = np.ones(len(means), dtype=bool)
# Exclude ±3 neighbours around peak
for i in range(max(0, idx_peak-3), min(len(means), idx_peak+4)):
    baseline_mask[i] = False
if baseline_mask.sum() > 0:
    baseline = means[baseline_mask].mean()
    n_sigma = (means[idx_peak] - baseline) / stats[idx_peak]['std'] if stats[idx_peak]['std'] > 0 else 0
    print(f"      Baseline (excl. peak±3): {baseline:.1f}")
    print(f"      Peak significance: {n_sigma:.1f}σ")
    if n_sigma > 3:
        print(f"      ✅ Peak is statistically significant (>{n_sigma:.0f}σ)")
    elif n_sigma > 2:
        print(f"      ⚠️  Peak is marginal ({n_sigma:.1f}σ)")
    else:
        print(f"      ❌ Peak is NOT significant ({n_sigma:.1f}σ) — likely noise")

# Save final output
final = {
    "experiment": f"χ_eff(T) L={L} Multi-Run Statistical Scan",
    "params": {
        "L": L, "N": N_total, "p_nuc": p_nuc,
        "eq_sweeps": eq_sweeps, "sample_sweeps": sample_sweeps,
        "T_start": T_start, "T_end": T_end, "T_step": T_step,
        "N_RUNS": N_RUNS, "SEED_BASE": SEED_BASE
    },
    "statistics": stats,
    "runs": all_runs,
    "completed_at": datetime.now().isoformat()
}

outpath = "output/vacuum/chi_eff_L64_multirun_final.json"
with open(outpath, "w") as f:
    json.dump(final, f, indent=2)

print(f"\n  ✅ Final output → {outpath}")
print(f"  ✅ Checkpoint → {CHECKPOINT}")
print(f"{'='*72}")
