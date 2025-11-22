# Adaptive Sampling Integration Summary

## Overview

Adaptive sampling has been successfully integrated into all forward model testing scripts. This allows for 10× better representation of blob regions compared to uniform random sampling.

## Files Updated

### 1. test_forward_model.py
**Purpose:** Test forward model with optimized parameters

**Changes:**
- Added adaptive sampling toggle: `use_adaptive = True`
- Added sampling ratio parameters:
  - `blob_ratio = 0.5` (50% of samples from blob interior)
  - `boundary_ratio = 0.3` (30% from boundaries)
  - Background = 20% (remaining)
- Updated `BIOQICDataLoader` call to pass adaptive sampling parameters
- Added informative print statements showing sampling strategy

**Location:** Lines 229-256

**Usage:**
```python
# Enable adaptive sampling
use_adaptive = True  # Set to False for uniform sampling

# Run test
python bioqic_pielm/test_forward_model.py
```

### 2. grid_search_forward_mu.py
**Purpose:** Compare loss functions (MSE, Sobolev, Correlation, Relative L2)

**Changes:**
- Added adaptive sampling parameters (same as above)
- Integrated with existing grid search over:
  - BC weights: [10, 100, 1000]
  - Neurons: [100, 500, 1000]
  - Points: 10,000 (fixed)

**Location:** Lines 328-356

**Usage:**
```python
# Enable/disable adaptive sampling
use_adaptive = True  # Toggle here

# Run comparison
python bioqic_pielm/grid_search_forward_mu.py
```

### 3. grid_sweep_forward.py
**Purpose:** Comprehensive grid sweep over sampling points, neurons, and BC weights

**Changes:**
- Added adaptive sampling parameters
- Applied to all combinations in grid sweep:
  - BC weights: [0, 1, 10, 100]
  - Neurons: [2000, 5000, 10000]
  - Sampling points: [1000, 10000, 50000]

**Location:** Lines 78-132

**Usage:**
```python
# Configure adaptive sampling
use_adaptive = True
blob_ratio = 0.5
boundary_ratio = 0.3

# Run grid sweep (36 combinations)
python bioqic_pielm/grid_sweep_forward.py
```

## Adaptive Sampling Configuration

All three scripts use consistent parameters:

```python
# ADAPTIVE SAMPLING PARAMETERS
use_adaptive = True              # Enable/disable adaptive sampling
blob_ratio = 0.5                 # 50% samples from blob interior
boundary_ratio = 0.3             # 30% samples from boundaries
# background_ratio = 0.2 (implicit)  # 20% samples from background
```

### Region Detection Parameters (in data_loader.py)

- **Blob threshold:** 8000 Pa (midpoint between 3kPa background and 10kPa blobs)
- **Boundary width:** 5mm (distance threshold for boundary classification)

## Expected Improvements

### Uniform Sampling (Baseline)
- Blob representation: ~280 samples (5.7%) from 5000 total
- Background: ~4720 samples (94.3%)
- **Problem:** Under-represents rare but important blob features

### Adaptive Sampling (New)
- Blob representation: ~2935 samples (58.7%) from 5000 total
- Background: ~2065 samples (41.3%)
- **Improvement:** 10× more blob samples for same total count

## Benefits by Use Case

### 1. test_forward_model.py
- Better displacement prediction in blob regions
- More accurate boundary gradient estimation
- Improved R² scores in heterogeneous regions

### 2. grid_search_forward_mu.py
- Enhanced loss function discrimination
- Better detection of stiffness heterogeneity
- More robust loss comparisons

### 3. grid_sweep_forward.py
- Consistent blob representation across all sampling levels
- Fair comparisons between 1k, 10k, 50k point configurations
- More reliable optimal parameter identification

## Comparison Mode

All scripts support easy A/B testing:

```python
# Test with uniform sampling
use_adaptive = False
# Run experiment...

# Test with adaptive sampling
use_adaptive = True
# Run experiment again...

# Compare results
```

## Validation

To verify adaptive sampling is working:

1. **Check console output** for region detection:
   ```
   Region detection:
     Blob interior: 970 points (1.2%)
     Boundaries: 11,890 points (14.9%)
     Background: 67,140 points (83.9%)
   Sample allocation:
     Blob samples: 2,500 (50.0%)
     Boundary samples: 1,500 (30.0%)
     Background samples: 1,000 (20.0%)
   ```

2. **Verify final distribution:**
   ```
   Sampled 5000 points
   Blob samples: 2935 (58.7%)
   Background samples: 2065 (41.3%)
   ```

3. **Compare performance metrics:**
   - R² score should improve in blob regions
   - MSE should decrease for heterogeneous predictions
   - Spatial error plots should show better blob coverage

## Implementation Details

### Core Logic (data_loader.py)

```python
class BIOQICDataLoader:
    def __init__(
        self,
        data_dir: str,
        subsample: int,
        adaptive_sampling: bool = False,  # NEW
        blob_sample_ratio: float = 0.5,   # NEW
        boundary_sample_ratio: float = 0.3 # NEW
    ):
        ...

    def _detect_blob_regions(coords, stiffness):
        # Classify points as blob/boundary/background
        # Uses KDTree for efficient spatial queries
        ...

    def _adaptive_subsample(coords, stiffness, target_count):
        # Stratified sampling from each region
        ...
```

### Integration Pattern

All test scripts follow this pattern:

```python
# 1. Configure adaptive sampling
use_adaptive = True
blob_ratio = 0.5
boundary_ratio = 0.3

# 2. Create loader with parameters
loader = BIOQICDataLoader(
    data_dir=data_dir,
    subsample=n_points,
    adaptive_sampling=use_adaptive,
    blob_sample_ratio=blob_ratio,
    boundary_sample_ratio=boundary_ratio
)

# 3. Load data (sampling happens here)
data = loader.load()

# 4. Use data as normal
coords = data['coords']
u_raw = data['u_raw']
mu_raw = data['mu_raw']
```

## Next Steps

1. **Run baseline comparisons:**
   ```bash
   # Uniform sampling
   python bioqic_pielm/test_forward_model.py  # use_adaptive=False

   # Adaptive sampling
   python bioqic_pielm/test_forward_model.py  # use_adaptive=True
   ```

2. **Compare metrics:**
   - R² scores
   - MSE in blob vs background regions
   - Spatial error distribution
   - Computational time (should be similar)

3. **Tune sampling ratios if needed:**
   - Try 70/20/10 (more blob-focused)
   - Try 40/40/20 (more boundary-focused)
   - Evaluate which works best for forward problem

## Documentation References

- **Implementation:** [data_loader.py](data_loader.py) lines 24-285
- **Usage guide:** [ADAPTIVE_SAMPLING_README.md](ADAPTIVE_SAMPLING_README.md)
- **Analysis:** [DISPLACEMENT_ANALYSIS.md](../DISPLACEMENT_ANALYSIS.md)
- **Standalone test:** [test_adaptive_direct.py](../test_adaptive_direct.py)

---

**Status:** ✓ All forward model test scripts updated
**Date:** 2025-01-22
**Integration:** Complete and ready for testing
