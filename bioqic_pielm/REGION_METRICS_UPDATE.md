# Region-Specific Error Metrics Update

## Overview

Both `test_forward_model.py` and `grid_search_forward_mu.py` have been updated to compute and display error metrics separately for:
- **Overall** (all points)
- **Blob region** (μ > 8 kPa)
- **Background region** (μ ≤ 8 kPa)

This allows you to see how well the forward model performs in heterogeneous (blob) vs homogeneous (background) regions.

## Changes Made

### 1. test_forward_model.py

**New function added:**
```python
def compute_region_metrics(u_pred, u_meas, mu_true, blob_threshold=8000.0):
    """Compute error metrics separately for blob and background regions."""
```

**Metrics computed per region:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Max Error
- Number of points

**Output format:**
```
============================================================
OVERALL METRICS
============================================================
  MSE: 5.280683e-07
  MAE: 4.969621e-04
  R²: 0.9922
  Max Error: 2.336894e-02
  Points: 10000

============================================================
BLOB REGION METRICS (μ > 8 kPa)
============================================================
  MSE: 3.142567e-07
  MAE: 3.245621e-04
  R²: 0.9945
  Max Error: 1.123456e-02
  Points: 2000 (20.0%)

============================================================
BACKGROUND REGION METRICS (μ ≤ 8 kPa)
============================================================
  MSE: 6.123456e-07
  MAE: 5.678912e-04
  R²: 0.9910
  Max Error: 2.336894e-02
  Points: 8000 (80.0%)
```

### 2. grid_search_forward_mu.py

**Updates:**

1. **Adaptive sampling ratios updated:**
   ```python
   blob_ratio = 0.2     # 20% blob samples (was 50%)
   boundary_ratio = 0.1 # 10% boundary samples (was 30%)
   # background = 70% (implicit)
   ```

2. **Region metrics added to CSV output:**
   - `blob_r2`: R² score in blob region
   - `blob_mse`: MSE in blob region
   - `background_r2`: R² score in background region
   - `background_mse`: MSE in background region

3. **Console output enhanced:**
   ```
   [1/18] bc_weight=10, neurons=100, mu=constant_5000
     Overall R²:  0.9922
     MSE:         5.280683e-07
     Relative L2: 1.123456e-03
     Sobolev:     2.345678e-03
     Correlation: 0.9961 (loss=3.900000e-03)
     Blob R²:     0.9945 (MSE=3.142567e-07)
     Background R²: 0.9910 (MSE=6.123456e-07)
   ```

## Key Benefits

### 1. Diagnostic Power
- Identify if forward model struggles more in blob or background regions
- Detect if adaptive sampling is actually improving blob region accuracy

### 2. Loss Function Comparison
- See which loss function (MSE, Sobolev, Correlation, Relative L2) performs best in blob regions
- Determine if a loss function that works well overall also works well in heterogeneous regions

### 3. Parameter Tuning
- Choose BC weights and neuron counts that optimize blob region accuracy
- Balance overall performance vs region-specific performance

## Expected Insights

Based on the displacement analysis (DISPLACEMENT_ANALYSIS.md), we expect:

1. **Blob regions should be easier to predict:**
   - Lower variance in displacement (7.6× smaller than background)
   - More uniform displacement field
   - Should see **higher R²** in blob regions

2. **Background regions may have higher error:**
   - Higher variance (more spread in displacement)
   - May see **lower R²** in background regions

3. **Adaptive sampling impact:**
   - With 20/10/70 split: 2000 blob samples vs ~600 in uniform sampling
   - Should improve blob R² compared to uniform sampling
   - Background R² may decrease slightly due to fewer samples

## Usage

### Test Forward Model
```bash
python bioqic_pielm/test_forward_model.py
```

Output will show region-specific metrics after overall metrics.

### Grid Search with Loss Function Comparison
```bash
python bioqic_pielm/grid_search_forward_mu.py
```

Results CSV will include columns:
- `r2`, `blob_r2`, `background_r2`
- `mse`, `blob_mse`, `background_mse`

## Interpreting Results

### Good Forward Model Performance:
- Overall R² > 0.95
- Blob R² ≥ Overall R² (blobs should be easier)
- Background R² close to Overall R² (within 0.05)

### Signs of Issues:
- Blob R² < Background R² (unexpected - blobs should be easier)
- Large gap between blob and background R² (> 0.15)
- Very low blob R² despite high overall R² (model ignoring heterogeneity)

### Adaptive Sampling Working:
- Blob R² improves compared to uniform sampling
- Blob MSE decreases
- Small decrease in background R² is acceptable trade-off

## Next Steps

1. **Run test_forward_model.py** with updated 20/10/70 sampling:
   - Compare blob vs background R²
   - Check if R² improved from previous 0.1746 overall

2. **Run grid_search_forward_mu.py**:
   - Identify which BC weight gives best blob R²
   - Check if Sobolev loss helps blob region accuracy
   - Compare constant vs heterogeneous mu discrimination by region

3. **Compare adaptive vs uniform:**
   - Run both scripts with `use_adaptive=False`
   - Compare blob R² between the two modes
   - Quantify improvement from adaptive sampling

## Files Modified

1. **bioqic_pielm/test_forward_model.py**
   - Lines 21-53: Added `compute_region_metrics()` function
   - Lines 56-88: Updated `test_forward_given_mu()` to include region metrics
   - Lines 119-153: Added region-specific print statements

2. **bioqic_pielm/grid_search_forward_mu.py**
   - Lines 30-64: Added `compute_region_metrics()` function
   - Lines 135-179: Updated `test_forward_config()` to include region metrics
   - Lines 385-388: Updated adaptive sampling ratios to 20/10/70
   - Lines 491-518: Added region metrics to console output and CSV

---

**Status:** ✓ Both scripts updated and ready to test
**Date:** 2025-01-22
**Update:** Region-specific error metrics for blob vs background analysis
