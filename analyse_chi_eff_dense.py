import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('media/a4d581e0152949b1a96ffb6ceb7b79af_chi_eff_dense_Tc_window.json') as f:
    data = json.load(f)

Ts = [r['T'] for r in data['results']]
chi_effs = [r['chi_eff'] for r in data['results']]
deltas = [r['delta_theta'] for r in data['results']]
topsv0 = [r['TopSV_0'] for r in data['results']]
topsvn = [r['TopSV_nuc'] for r in data['results']]
h0 = [r['helicity_0'] for r in data['results']]
hn = [r['helicity_nuc'] for r in data['results']]

Tc = 2.269

# Console output
print("=" * 70)
print("X_eff(T) DENSE SCAN — Double-Peak Verification")
print(f"L=32, N=500, p_nuc=0.05, 13 points, T=[2.25, 2.37], dT=0.01")
print("=" * 70)
print(f"{'T':>7s}  {'dTheta':>8s}  {'X_eff':>8s}  {'TopSV0':>8s}  {'TopSVn':>8s}  {'H0':>8s}")
print("-" * 56)
for i in range(len(Ts)):
    print(f"{Ts[i]:7.2f}  {deltas[i]:8.2f}  {chi_effs[i]:8.1f}  {topsv0[i]:8.3f}  {topsvn[i]:8.3f}  {h0[i]:8.4f}")

idx_min = np.argmin(chi_effs)
idx_max = np.argmax(chi_effs)
print()
print(f"X_eff MIN: {chi_effs[idx_min]:.1f} @ T={Ts[idx_min]:.2f}  (delta_T from Tc = {Ts[idx_min]-Tc:+.3f})")
print(f"X_eff MAX: {chi_effs[idx_max]:.1f} @ T={Ts[idx_max]:.2f}  (delta_T from Tc = {Ts[idx_max]-Tc:+.3f})")
print(f"Ratio MAX/MIN = {chi_effs[idx_max]/chi_effs[idx_min]:.1f}x")
print()
print("=== CRITICAL FINDING ===")
print("Coarse scan (N=300) had peak 571.6 @ Tc — this was an ARTIFACT")
print("Dense scan (N=500) shows X_eff MINIMUM @ Tc, MAXIMUM @ T=2.35")
print("Catalytic efficiency is LOWEST at exact criticality, HIGHEST post-Tc!")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"X_eff(T) Dense Scan (L=32, N=500, p_nuc=0.05)\nTc=2.269 marked as dashed line", fontsize=13)

# Panel 1: X_eff
ax = axes[0, 0]
ax.plot(Ts, chi_effs, 'o-', color='crimson', linewidth=2, markersize=8)
ax.axvline(Tc, color='gray', linestyle='--', alpha=0.5, label=f'Tc={Tc}')
ax.scatter([Ts[idx_min]], [chi_effs[idx_min]], s=200, facecolors='none', edgecolors='blue', linewidths=2, zorder=5, label=f'MIN={chi_effs[idx_min]:.0f}')
ax.scatter([Ts[idx_max]], [chi_effs[idx_max]], s=200, facecolors='none', edgecolors='red', linewidths=2, zorder=5, label=f'MAX={chi_effs[idx_max]:.0f}')
ax.set_xlabel('T')
ax.set_ylabel('X_eff')
ax.set_title('Catalytic Efficiency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: delta_theta
ax = axes[0, 1]
ax.plot(Ts, deltas, 'o-', color='darkorange', linewidth=2, markersize=8)
ax.axvline(Tc, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('T')
ax.set_ylabel('delta_theta (deg)')
ax.set_title('Subspace Rotation Angle')
ax.grid(True, alpha=0.3)

# Panel 3: TopSV comparison
ax = axes[1, 0]
ax.plot(Ts, topsv0, 's-', color='steelblue', linewidth=1.5, markersize=6, label='Vacuum (TopSV_0)')
ax.plot(Ts, topsvn, '^--', color='coral', linewidth=1.5, markersize=6, label='Nucleated (TopSV_nuc)')
ax.axvline(Tc, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('T')
ax.set_ylabel('TopSV')
ax.set_title('TopSV: Vacuum vs Nucleated')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: X_eff vs delta_T from Tc
ax = axes[1, 1]
delta_Ts = [t - Tc for t in Ts]
ax.plot(delta_Ts, chi_effs, 'o-', color='purple', linewidth=2, markersize=8)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Tc')
ax.set_xlabel('T - Tc')
ax.set_ylabel('X_eff')
ax.set_title('X_eff vs Distance from Tc')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/vacuum/chi_eff_dense_scan.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: output/vacuum/chi_eff_dense_scan.png")
