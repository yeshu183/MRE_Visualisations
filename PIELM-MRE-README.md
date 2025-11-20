# PIELM-MRE Implementation Summary

## ✅ Completed Components

### 1. **Data Loading & Preprocessing** (`01_data_loading_phase1.ipynb`)
- ✅ Loaded BIOQIC four_target_phantom.mat (5D displacement array)
- ✅ Extracted 60 Hz frequency data
- ✅ Generated coordinate grid (80,000 points)
- ✅ Created ground truth stiffness via geometric segmentation
  - Background: 3 kPa
  - 4 targets: 10 kPa (radii: 10, 5, 3, 2 mm)
  - Voigt model: μ = μ' + iωη
- ✅ Saved preprocessed data to `data/processed/phase1_box/`

### 2. **Physics Module** (`physics_module.py`)
- ✅ JAX-based automatic differentiation
- ✅ Helmholtz equation implementations
- ✅ Physics loss functions
- ✅ Tested successfully ✓

### 3. **PIELM-MRE Core** (`pielm_mre.py`)
**Architecture:** Iterative Dual-Network ELM

- `PIELMFeatures`: Random features with analytical derivatives
- `PIELMNetwork`: Complex-valued ELM with ridge regression
- `IterativePIELMMRE`: Alternating optimization + curriculum learning

### 4. **Training Pipeline** (`train_phase1_pielm.py`)
- Complete end-to-end training script
- Evaluation and visualization

## 📐 Mathematical Framework

**MRE Inverse Problem:** Given u(x), find μ(x)

**PDE:** ∇·[μ(x)∇u(x)] + ρω²u(x) = 0

**PIELM Strategy:** Iterative alternating optimization with curriculum learning

## 🚀 Next Steps

Run training:
```bash
python train_phase1_pielm.py
```

**Status:** Implementation complete, ready for testing

## Overview
This README presents a comprehensive methodology for using Physics-Informed Extreme Learning Machine (PIELM) variants to solve the Magnetic Resonance Elastography (MRE) inverse problem in the liver. It summarizes relevant physics, dataset specifications, state-of-the-art research, and a phased implementation roadmap.

---

## Background

MRE aims to non-invasively map spatial stiffness variations in tissues by solving the Helmholtz equation:

\[
\nabla \cdot [\mu^*(\mathbf{r}) \nabla \mathbf{u}(\mathbf{r})] + \rho \omega^2 \mathbf{u}(\mathbf{r}) = 0
\]

where:
- \(\mathbf{u}(\mathbf{r})\): measured displacement field
- \(\mu^*(\mathbf{r}) = \mu'(\mathbf{r}) + i\mu''(\mathbf{r})\): complex shear modulus (unknown)
- \(\omega\), \(\rho\): excitation frequency, tissue density

The inverse problem estimates \(\mu^*\) from \(\mathbf{u}\).

---

## Selected Datasets

### ScienceDB (Phantom, Liver, Brain)
- **Spatial grids:** ~100x100x30 voxels
- **Displacement:** 3 components, multi-frequency (30-60 Hz), 4 offsets
- **Modulus maps:** available for simulated data

### BIOQIC (Simulated & Real)
- **FEM simulations:** Abdomen, brain, box; 50k-500k nodes; 3D, 8 timesteps
- **Real Liver:** 64-128x64-128x10-20 voxels; multiple subjects

---

## PIELM Methodology (With Heterogeneous Physics)

### Step 1: Formulate Problem
- Use full heterogeneous Helmholtz equation
- Input: spatial coordinates, displacement
- Output: spatial modulus map \(\mu^*(\mathbf{r})\)

### Step 2: Network Architecture
- **Single hidden layer, fixed random weights (ELM/PIELM)**
- Output weights solved by direct least squares
- RBF activation for spatial localization
- Input features: (x, y, z[, frequency])

### Step 3: Physics-Informed Loss
- **PDE residual:** Enforce Helmholtz equation
- **Boundary condition loss:** Stress-free or specified
- **Data matching (where simulated ground truth is available)**
- **Regularization:** Physically plausible \(\mu^*\)

### Step 4: Implementation Protocol

1. **Preprocess Datasets:** Normalize spatial grids, extract displacements, sample collocation points.
2. **Iterative PIELM Training:** Alternate updates; refine modulus estimates to minimize PDE residuals.
3. **Curriculum Learning:** Linearly increase PDE complexity (homogeneous \(\to\) heterogeneous).
4. **Multi-Frequency Integration:** Stack as channels or ensemble networks separately for each frequency.

---

## Suggested Training and Validation Schedule

| Stage | Dataset | Goal |
|-------|---------|------|
| 1 | BIOQIC FEM Box | Validate implementation |
| 2 | FEM Abdomen | Learn on smooth/heterogeneous \(\mu^*\) |
| 3 | SciDB Phantom | Test noise/multi-frequency |
| 4 | SciDB Liver/BIOQIC Real | Real data validation |

**Metrics:** MAE, relative L2 error, SSIM, clinical value ranges, cross-frequency consistency

---

## Key Insights from Literature
- PIELM achieves training speeds 10-100x faster than PINNs
- Iterative/curriculum learning PIELM variants handle nonlinear, sharp boundary, and noisy inverse problems
- PINN-based MRE inversions for liver (NIH 2023) show full heterogeneous PDE provides best elastogram accuracy
- No published PIELM-based MRE inversion yet—your project fills this gap

---

## Implementation Notes
- Downsample or crop for computational feasibility in first experiments
- Use regularization for physically plausible modulus
- Always evaluate on ground truth where available and compare quantitatively
- Use multi-frequency data for more robust inversion (joint loss or fusion)

---

## References
- Main papers: PINN-MRE liver inversion (NIH 2023), Iterative PIELM, Curriculum-Driven PIELM, Eig-PIELM
- Datasets: ScienceDB (Yuan Feng et al., 2025), BIOQIC (Charite, Ariyurek UM-RAM)
- Open-source code: Blue-Giant/PIELM_Numpy

---

## Next Steps
- Prototype PIELM on simple simulated box dataset
- Progressively scale to more challenging, anatomically realistic datasets
- Benchmark against PINN and direct inversion; optimize for accuracy & speed
- Prepare code & documentation for publication

For code templates, loss functions, or data preprocessing examples, see accompanying scripts or ask for further breakdowns.
