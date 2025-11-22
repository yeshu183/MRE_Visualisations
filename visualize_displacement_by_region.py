"""
Visualize displacement distributions in background vs blob regions.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
data_dir = Path("data/processed/phase1_box")
coords = np.load(data_dir / "coordinates.npy")
displacement = np.load(data_dir / "displacement.npy")
stiffness = np.load(data_dir / "stiffness_ground_truth.npy")

print("="*80)
print("DISPLACEMENT DISTRIBUTION BY REGION")
print("="*80)

# Extract z-component (real part)
if np.iscomplexobj(displacement):
    u_z = displacement[:, 2].real
else:
    u_z = displacement[:, 2]

# Classify regions
mu_real = stiffness.real.flatten()  # Flatten to 1D
is_blob = (mu_real > 8000.0).astype(bool)

blob_u = u_z[is_blob]
background_u = u_z[is_blob == False]

print(f"\nTotal points: {len(coords):,}")
print(f"Blob points: {len(blob_u):,} ({100*len(blob_u)/len(coords):.1f}%)")
print(f"Background points: {len(background_u):,} ({100*len(background_u)/len(coords):.1f}%)")

print(f"\nDisplacement statistics:")
print(f"  Overall: mean={u_z.mean():.6f}, std={u_z.std():.6f}, range=[{u_z.min():.6f}, {u_z.max():.6f}]")
print(f"  Blob:    mean={blob_u.mean():.6f}, std={blob_u.std():.6f}, range=[{blob_u.min():.6f}, {blob_u.max():.6f}]")
print(f"  Background: mean={background_u.mean():.6f}, std={background_u.std():.6f}, range=[{background_u.min():.6f}, {background_u.max():.6f}]")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histograms comparison
ax = axes[0, 0]
bins = np.linspace(u_z.min(), u_z.max(), 100)
ax.hist(background_u, bins=bins, alpha=0.6, label=f'Background (μ≤8kPa, n={len(background_u):,})',
        color='steelblue', density=True)
ax.hist(blob_u, bins=bins, alpha=0.6, label=f'Blob (μ>8kPa, n={len(blob_u):,})',
        color='orangered', density=True)
ax.set_xlabel('Displacement u_z (m)')
ax.set_ylabel('Probability Density')
ax.set_title('Displacement Distribution by Region')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Box plots
ax = axes[0, 1]
data_to_plot = [background_u, blob_u]
bp = ax.boxplot(data_to_plot, labels=['Background\n(μ≤8kPa)', 'Blob\n(μ>8kPa)'],
                patch_artist=True, showfliers=False)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('orangered')
ax.set_ylabel('Displacement u_z (m)')
ax.set_title('Displacement Range by Region')
ax.grid(True, alpha=0.3, axis='y')

# Add mean markers
means = [background_u.mean(), blob_u.mean()]
ax.plot([1, 2], means, 'D', color='black', markersize=8, label='Mean', zorder=3)
ax.legend()

# 3. Cumulative distribution
ax = axes[1, 0]
ax.hist(background_u, bins=bins, alpha=0.6, cumulative=True, density=True,
        label='Background', color='steelblue', histtype='step', linewidth=2)
ax.hist(blob_u, bins=bins, alpha=0.6, cumulative=True, density=True,
        label='Blob', color='orangered', histtype='step', linewidth=2)
ax.set_xlabel('Displacement u_z (m)')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Cumulative Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Absolute displacement comparison
ax = axes[1, 1]
bins_abs = np.linspace(0, max(np.abs(blob_u).max(), np.abs(background_u).max()), 100)
ax.hist(np.abs(background_u), bins=bins_abs, alpha=0.6,
        label=f'Background |u_z|', color='steelblue', density=True)
ax.hist(np.abs(blob_u), bins=bins_abs, alpha=0.6,
        label=f'Blob |u_z|', color='orangered', density=True)
ax.set_xlabel('Absolute Displacement |u_z| (m)')
ax.set_ylabel('Probability Density')
ax.set_title('Displacement Magnitude Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics text
stats_text = (
    f"Background: μ={background_u.mean():.5f}m, σ={background_u.std():.5f}m\n"
    f"Blob:       μ={blob_u.mean():.5f}m, σ={blob_u.std():.5f}m\n"
    f"Ratio (blob/background): {blob_u.std()/background_u.std():.3f}×"
)
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig('displacement_distribution_by_region.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: displacement_distribution_by_region.png")

# Additional analysis: spatial plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Take middle z-slice
z_mid = coords[:, 2].mean()
z_tol = 0.002
mask = np.abs(coords[:, 2] - z_mid) < z_tol

x_slice = coords[mask, 0]
y_slice = coords[mask, 1]
u_slice = u_z[mask]
mu_slice = mu_real[mask]

# 1. Displacement field
ax = axes[0]
sc = ax.scatter(x_slice, y_slice, c=u_slice, s=15, cmap='seismic',
                vmin=-np.abs(u_slice).max(), vmax=np.abs(u_slice).max())
plt.colorbar(sc, ax=ax, label='u_z (m)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Displacement Field u_z(x,y)')
ax.set_aspect('equal')

# 2. Stiffness field
ax = axes[1]
sc = ax.scatter(x_slice, y_slice, c=mu_slice, s=15, cmap='jet', vmin=3000, vmax=10000)
plt.colorbar(sc, ax=ax, label='μ (Pa)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Stiffness Field μ(x,y)')
ax.set_aspect('equal')

# 3. Overlay: displacement magnitude with blob boundaries
ax = axes[2]
is_blob_slice = (mu_slice > 8000).astype(bool)
background_mask = is_blob_slice == False
blob_mask = is_blob_slice

# Plot background points
sc1 = ax.scatter(x_slice[background_mask], y_slice[background_mask],
                 c=np.abs(u_slice[background_mask]), s=15, cmap='Blues',
                 vmin=0, vmax=np.abs(u_slice).max(), alpha=0.5, label='Background')

# Plot blob points
sc2 = ax.scatter(x_slice[blob_mask], y_slice[blob_mask],
                 c=np.abs(u_slice[blob_mask]), s=15, cmap='Reds',
                 vmin=0, vmax=np.abs(u_slice).max(), alpha=0.8, label='Blob')

plt.colorbar(sc2, ax=ax, label='|u_z| (m)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Displacement Magnitude by Region')
ax.set_aspect('equal')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('displacement_spatial_by_region.png', dpi=150, bbox_inches='tight')
print(f"Saved: displacement_spatial_by_region.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Statistical test
from scipy import stats

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(background_u, blob_u)
print(f"\nTwo-sample t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4e}")

# Kolmogorov-Smirnov test (for distribution differences)
ks_stat, ks_p = stats.ks_2samp(background_u, blob_u)
print(f"\nKolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_p:.4e}")

# Variance ratio test
var_ratio = blob_u.var() / background_u.var()
print(f"\nVariance ratio (blob/background): {var_ratio:.4f}")

if p_value < 0.001:
    print(f"\n** Displacement distributions are SIGNIFICANTLY different (p < 0.001)")
else:
    print(f"\nNo significant difference in distributions (p = {p_value:.4e})")

plt.show()
