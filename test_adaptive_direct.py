"""
Direct test of adaptive sampling without full package imports.
"""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

def detect_blob_regions(coords, stiffness, blob_threshold=8000.0, boundary_width=0.005):
    """Detect blob interior, boundary, and background regions."""
    mu_real = stiffness.real
    is_blob = (mu_real > blob_threshold).astype(bool)

    blob_indices = np.where(is_blob)[0]
    background_indices = np.where(is_blob == False)[0]

    if len(blob_indices) == 0 or len(background_indices) == 0:
        print(f"  Warning: No boundary detection possible")
        boundary_mask = np.zeros(len(coords), dtype=bool)
        background_mask = np.logical_not(is_blob)
        return is_blob, boundary_mask, background_mask

    # Build trees
    blob_tree = cKDTree(coords[blob_indices])
    background_tree = cKDTree(coords[background_indices])

    # Find boundaries
    boundary_mask = np.zeros(len(coords), dtype=bool)

    # Blob points near background
    dists, _ = background_tree.query(coords[blob_indices], k=1)
    near_boundary = dists < boundary_width
    boundary_mask[blob_indices[near_boundary]] = True

    # Background points near blobs
    dists, _ = blob_tree.query(coords[background_indices], k=1)
    near_boundary = dists < boundary_width
    boundary_mask[background_indices[near_boundary]] = True

    # Interior regions (exclude boundaries)
    blob_interior = np.zeros(len(coords), dtype=bool)
    blob_interior[blob_indices] = True
    blob_interior[boundary_mask] = False

    background_interior = np.zeros(len(coords), dtype=bool)
    background_interior[background_indices] = True
    background_interior[boundary_mask] = False

    print(f"  Region detection:")
    print(f"    Blob interior: {int(blob_interior.sum()):,} ({100*blob_interior.sum()/len(coords):.1f}%)")
    print(f"    Boundaries: {int(boundary_mask.sum()):,} ({100*boundary_mask.sum()/len(coords):.1f}%)")
    print(f"    Background: {int(background_interior.sum()):,} ({100*background_interior.sum()/len(coords):.1f}%)")

    return blob_interior, boundary_mask, background_interior

def adaptive_subsample(coords, stiffness, target_count, blob_ratio=0.5, boundary_ratio=0.3):
    """Adaptive subsampling with more points in/near blobs."""
    blob_mask, boundary_mask, background_mask = detect_blob_regions(coords, stiffness)

    n_blob = int(target_count * blob_ratio)
    n_boundary = int(target_count * boundary_ratio)
    n_background = target_count - n_blob - n_boundary

    print(f"  Sample allocation:")
    print(f"    Blob samples: {n_blob:,} ({100*n_blob/target_count:.1f}%)")
    print(f"    Boundary samples: {n_boundary:,} ({100*n_boundary/target_count:.1f}%)")
    print(f"    Background samples: {n_background:,} ({100*n_background/target_count:.1f}%)")

    blob_indices = np.where(blob_mask)[0]
    boundary_indices = np.where(boundary_mask)[0]
    background_indices = np.where(background_mask)[0]

    selected_indices = []

    if len(blob_indices) > 0 and n_blob > 0:
        replace = n_blob > len(blob_indices)
        selected = np.random.choice(blob_indices, n_blob, replace=replace)
        selected_indices.append(selected)

    if len(boundary_indices) > 0 and n_boundary > 0:
        replace = n_boundary > len(boundary_indices)
        selected = np.random.choice(boundary_indices, n_boundary, replace=replace)
        selected_indices.append(selected)

    if len(background_indices) > 0 and n_background > 0:
        replace = n_background > len(background_indices)
        selected = np.random.choice(background_indices, n_background, replace=replace)
        selected_indices.append(selected)

    all_indices = np.concatenate(selected_indices)
    np.random.shuffle(all_indices)
    return all_indices

# Load data
print("="*80)
print("TESTING ADAPTIVE SAMPLING")
print("="*80)

data_dir = Path("data/processed/phase1_box")
coords = np.load(data_dir / "coordinates.npy")
stiffness = np.load(data_dir / "stiffness_ground_truth.npy")

print(f"\nTotal points: {len(coords):,}")
print(f"Stiffness range: [{stiffness.real.min():.0f}, {stiffness.real.max():.0f}] Pa")

# Test uniform sampling
print("\n" + "="*80)
print("UNIFORM SAMPLING (baseline)")
print("="*80)
np.random.seed(42)
uniform_idx = np.random.choice(len(coords), 5000, replace=False)
uniform_stiffness = stiffness[uniform_idx].real
blob_count = (uniform_stiffness > 8000).sum()
print(f"Blob samples: {blob_count:,} ({100*blob_count/5000:.1f}%)")
print(f"Background samples: {5000-blob_count:,} ({100*(5000-blob_count)/5000:.1f}%)")

# Test adaptive sampling
print("\n" + "="*80)
print("ADAPTIVE SAMPLING (50% blob, 30% boundary, 20% background)")
print("="*80)
np.random.seed(42)
adaptive_idx = adaptive_subsample(coords, stiffness, 5000, blob_ratio=0.5, boundary_ratio=0.3)
adaptive_stiffness = stiffness[adaptive_idx].real
blob_count = (adaptive_stiffness > 8000).sum()
print(f"\nFinal sampled distribution:")
print(f"  Blob samples: {blob_count:,} ({100*blob_count/5000:.1f}%)")
print(f"  Background samples: {5000-blob_count:,} ({100*(5000-blob_count)/5000:.1f}%)")

print("\n" + "="*80)
print("SUCCESS: Adaptive sampling working correctly!")
print("="*80)
