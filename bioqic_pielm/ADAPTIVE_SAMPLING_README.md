# Adaptive Sampling for Forward MRE Problem

## Overview

Adaptive sampling concentrates training points in regions of interest (blobs and boundaries) instead of uniform random sampling across the entire domain. This improves forward problem accuracy by providing more data where stiffness gradients are highest.

## Implementation

### Data Loader Changes

**File:** `data_loader.py`

Added three new parameters to `BIOQICDataLoader.__init__()`:
- `adaptive_sampling` (bool): Enable adaptive sampling (default: False)
- `blob_sample_ratio` (float): Fraction of samples inside blobs (default: 0.5)
- `boundary_sample_ratio` (float): Fraction of samples near blob boundaries (default: 0.3)
- Remaining fraction goes to background region

### Region Detection Method

`_detect_blob_regions(coords, stiffness, blob_threshold=8000.0, boundary_width=0.005)`

**Algorithm:**
1. Classify points as blob (μ > 8000 Pa) or background (μ ≤ 8000 Pa)
2. Build separate KDTree for blob points and background points
3. For each blob point, find distance to nearest background point
4. For each background point, find distance to nearest blob point
5. Mark points as boundary if distance < 5mm (boundary_width)
6. Compute interior masks (blob/background excluding boundaries)

**Memory optimization:** Uses separate trees and nearest-neighbor queries instead of all-pairs distance matrix to avoid O(N²) memory

### Adaptive Subsampling Method

`_adaptive_subsample(coords, stiffness, target_count)`

**Algorithm:**
1. Detect blob interior, boundaries, and background regions
2. Allocate samples:
   - 50% from blob interior (high stiffness regions)
   - 30% from boundaries (high gradient regions)
   - 20% from background (low priority)
3. Randomly sample from each region (with replacement if needed)
4. Shuffle combined samples to mix regions

## Usage

### In train.py

Add to experiment configuration:
```python
'forward_adaptive': {
    'description': 'Forward problem with adaptive sampling near blobs',
    'displacement_mode': 'z_component',
    'bc_strategy': 'box',
    'rho_omega2': None,
    'n_wave_neurons': 1000,
    'omega_basis': 170.0,
    'mu_range': (3000.0, 10000.0),
    'bc_weight': 10.0,
    'data_weight': 10.0,  # Strong data fitting
    'tv_weight': 0.0,
    'use_cnn_mu': False,
    'basis_type': 'sin',
    'iterations': 5000,
    'lr': 0.001,
    'loss_type': 'mse',
    'adaptive_sampling': True,      # ENABLE adaptive sampling
    'blob_sample_ratio': 0.5,       # 50% blob samples
    'boundary_sample_ratio': 0.3,   # 30% boundary samples
}
```

### Running Experiments

```bash
python bioqic_pielm/train.py --experiment forward_adaptive --subsample 5000
```

## Expected Results

### Uniform Sampling (Baseline)
With 5000 samples from 80,000 total points:
- Blob samples: ~280 (5.7%) - matches natural blob fraction
- Background samples: ~4720 (94.3%)

### Adaptive Sampling
With 5000 samples:
- Blob interior: ~2500 (50.0%)
- Boundaries: ~1500 (30.0%)
- Background: ~1000 (20.0%)
- **Total blob+boundary: ~2900 (58%)** vs ~280 (5.7%) in uniform

This gives **10× more samples** in regions of interest!

## Benefits for Forward Problem

1. **Better gradient estimation:** More points at blob boundaries where ∇μ is large
2. **Improved blob representation:** More points inside blobs to capture uniform μ=10kPa
3. **Reduced wasted sampling:** Fewer redundant points in uniform background (μ=3kPa)
4. **Same computational cost:** Still 5000 total samples, just better distributed

## Testing

Standalone test (no torch required):
```bash
python test_adaptive_direct.py
```

Full data loader test (requires torch environment):
```bash
python bioqic_pielm/test_adaptive_sampling.py
```

## Next Steps

1. Add `forward_adaptive` experiment to `EXPERIMENTS` dict in train.py
2. Run forward problem comparison:
   - Uniform sampling (baseline)
   - Adaptive sampling (50/30/20 split)
   - Adaptive sampling (70/20/10 split - even more blob focus)
3. Compare displacement prediction accuracy in blob regions
4. Visualize sample distribution in spatial μ(x,y) plots

## Parameters to Tune

- `blob_sample_ratio`: Higher = more blob interior samples
- `boundary_sample_ratio`: Higher = more boundary samples
- `blob_threshold`: Default 8000 Pa (midpoint between 3000 and 10000)
- `boundary_width`: Default 5mm (typical voxel spacing)

## Known Issues

- Requires scipy for cKDTree (already in dependencies)
- Memory efficient but still O(N log N) for tree construction
- If blobs are very small (< target samples), will sample with replacement
