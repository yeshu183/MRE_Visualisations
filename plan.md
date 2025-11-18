# PIELM-MRE Implementation Plan

## Project Overview
Implementation of Physics-Informed Extreme Learning Machine (PIELM) for solving the Magnetic Resonance Elastography (MRE) inverse problem in liver tissue. The approach uses **Iterative PIELM with Curriculum Learning** to progressively handle increasing complexity from simulated to real data.

---

## Architecture Choice: Iterative PIELM + Curriculum Learning

### Why Iterative PIELM?
- **Core Requirement:** The MRE inverse problem is coupled—we need to find stiffness (μ) that explains displacement (u), but u depends on μ.
- **Solution:** Iterative alternating optimization:
  1. Fix μ, solve for u (displacement network)
  2. Fix u, solve for μ (modulus network)
  3. Repeat until convergence

### Why Curriculum Learning?
- Start with simplified physics (homogeneous Helmholtz)
- Gradually introduce heterogeneity terms (∇μ)
- Prevents local minima and guides toward global solution
- Aligns with our "simple → complex" roadmap

### Why NOT Bayesian PIELM (for now)?
- Uncertainty quantification is valuable for clinical deployment
- But adds mathematical complexity that doesn't improve convergence
- **Decision:** Save for Version 2.0 after core solver is validated

---

## Three-Part Implementation Structure

### Part 1: Data Imports and Preprocessing
- Load MATLAB `.mat` files from ScienceDB and BIOQIC
- Extract displacement fields, magnitude images, masks
- Normalize spatial coordinates and physical units
- Generate collocation points for physics loss

### Part 2: Physics PDE Equations
- Implement Helmholtz equation residual calculator
- Support both homogeneous and heterogeneous formulations
- Compute spatial derivatives (∇u, ∇²u, ∇μ)
- Define boundary conditions (stress-free or specified)

### Part 3: Architecture and Methodology
- Build dual ELM networks (Displacement + Modulus)
- Implement iterative training loop
- Add curriculum learning scheduler
- Define combined loss function (Data + Physics + Regularization)

---

## Phased Implementation Roadmap

### Phase 1: Sanity Check (Homogeneous / Forward Problem)
**Goal:** Validate that ELM can fit displacement data and satisfy basic PDE

#### Dataset: BIOQIC FEM Box Simulation
- **Source:** `four_target_phantom.mat` (8 timesteps available)
- **Why:** 
  - Simple geometry (rectangular box)
  - Known ground truth stiffness
  - Multiple inclusions (4 targets with different stiffness)
  - Clean FEM data (no noise)
- **Specifications:**
  - FEM simulation with prescribed stiffness values
  - 8 timesteps for temporal analysis
  - Description PDF available for full specs

#### Physics:
- **Homogeneous Helmholtz:** μ∇²u + ρω²u = 0
- Assume constant μ initially, just fit displacement field

#### Architecture:
- **Input:** (x, y, z) coordinates
- **Hidden Layer:** Fixed random weights, Sine/Tanh activation
- **Output:** (u_real, u_imag) - complex displacement
- **Optimization:** Single-step Least Squares (Moore-Penrose pseudoinverse)

#### Success Metrics:
- Low data matching error (MSE between predicted and true u)
- PDE residual < threshold
- Visual inspection: smooth displacement field

---

### Phase 2: Coupled Solver (Heterogeneous / Inverse Problem - Simulated)

**Goal:** Solve for μ(x) - the actual inverse problem

#### Dataset Option A: BIOQIC FEM Abdomen Simulation ✓ RECOMMENDED
- **Source:** `AbdomenMRESimData.mat` (8 timesteps available)
- **Why:**
  - Anatomically realistic (includes liver region)
  - Heterogeneous stiffness distribution
  - Known ground truth for validation
  - Large-scale FEM (~50k-500k nodes mentioned in README)
  - Direct transition to Phase 3 real liver data
- **Data Contents:**
  - Displacement fields (3 components)
  - Prescribed stiffness (ground truth μ)
  - Tissue geometry/mask

#### Dataset Option B: ScienceDB Agar Phantom
- **Source:** `phantom/PhaseRaw_phantom.mat`, `phantom/U_phantom.mat`, `phantom/G_phantom.mat`
- **Why:**
  - Real MRI acquisition (not simulation)
  - Multi-frequency (30, 40, 50, 60 Hz)
  - Known ground truth (agar with 2 inclusions: 1 hard, 1 soft)
  - Dimensions: [x, y, z, phase offset, displacement direction, frequency]
- **Advantages:**
  - Tests robustness to real MRI noise
  - Multi-frequency data for joint inversion
- **Disadvantages:**
  - Less anatomically relevant than abdomen simulation
  - Smaller grid size (~100×100×30)

**Decision:** Start with **BIOQIC Abdomen Simulation** (cleaner, larger, anatomically relevant), then validate on **ScienceDB Phantom** (tests noise robustness).

#### Physics:
- **Full Heterogeneous Helmholtz:** ∇·[μ(x)∇u] + ρω²u = 0
- Expand: μ∇²u + ∇μ·∇u + ρω²u = 0

#### Architecture (Iterative):
- **Network A (Displacement):** Input (x,y,z) → Output u(x,y,z)
- **Network B (Modulus):** Input (x,y,z) → Output μ(x,y,z)
- **Training Loop:**
  ```
  for iteration in range(max_iter):
      # Step 1: Fix μ, train u network
      L_data = ||u_pred - u_measured||²
      L_physics = ||PDE_residual(u_pred, μ_fixed)||²
      Optimize weights_u
      
      # Step 2: Fix u, train μ network
      L_physics = ||PDE_residual(u_fixed, μ_pred)||²
      L_reg = ||∇μ||² (smoothness prior)
      Optimize weights_μ
  ```

#### Curriculum Learning Schedule:
- **Stage 1 (Epochs 1-100):** High regularization (λ_reg = 1.0), effectively near-homogeneous
- **Stage 2 (Epochs 101-300):** Reduce regularization (λ_reg = 0.1)
- **Stage 3 (Epochs 301+):** Full heterogeneity (λ_reg = 0.01)

#### Success Metrics:
- Mean Absolute Error (MAE) vs ground truth μ
- Relative L2 error: ||μ_pred - μ_true||/||μ_true||
- SSIM (Structural Similarity Index)
- Visual comparison: predicted vs ground truth elastograms

---

### Phase 3: Robust Solver (Real Data - Liver)

**Goal:** Handle noise, irregular boundaries, and clinical data

#### Dataset: ScienceDB Real Liver MRE ✓ PRIMARY TARGET
- **Source:** `liver/PhaseRaw_liver.mat`, `liver/U_liver.mat`, `liver/Mag_liver.mat`, `liver/G_liver.mat`
- **Why:**
  - Real human liver data (healthy volunteer)
  - Multi-frequency acquisition (30, 40, 50, 60 Hz)
  - Ground truth stiffness available (G_liver.mat - complex shear modulus)
  - Clinical relevance (our ultimate goal)
- **Specifications:**
  - Dimensions: [x, y, z, phase offset (4), displacement direction (3), frequency (4)]
  - PhaseRaw: Raw phase data in radians
  - U: Displacement field (radians)
  - Mag: Magnitude images
  - G: Complex shear modulus (Pa) - Real part: storage modulus, Imaginary part: loss modulus
  - Total: 8 phase offsets (4×2 for positive/negative encoding)

#### Alternative: BIOQIC Real Liver (if ScienceDB insufficient)
- **Source:** `MMRE_liver.mat`
- **Status:** Multi-frequency real liver data
- **Use Case:** Backup or additional validation dataset

#### Methodology Enhancements:
1. **Noise Handling:**
   - Add Gaussian noise augmentation during training
   - Increase data loss weight vs physics loss
   - Ensemble multiple network initializations

2. **Curriculum Learning (Advanced):**
   - Start with single frequency (60 Hz - highest signal)
   - Gradually add lower frequencies (50 → 40 → 30 Hz)
   - Final stage: Joint multi-frequency inversion

3. **Boundary Handling:**
   - Extract liver mask from Magnitude images
   - Apply soft boundary constraints (gradual loss weighting at edges)
   - Use stress-free boundary conditions where appropriate

4. **Multi-Frequency Strategy:**
   - **Option A:** Stack frequencies as input channels [x,y,z,f] → [μ_real, μ_imag]
   - **Option B:** Train separate networks per frequency, ensemble predictions
   - **Recommended:** Option A for end-to-end learning

#### Success Metrics:
- MAE/SSIM vs ground truth (G_liver.mat)
- Clinical plausibility: μ in range 2-4 kPa for healthy liver
- Cross-frequency consistency: predictions should agree across 30-60 Hz
- Visual assessment: smooth elastograms without artifacts

---

## Dataset Summary Table

| Phase | Dataset | Source | Type | Ground Truth? | Complexity | Priority |
|-------|---------|--------|------|---------------|------------|----------|
| 1 | FEM Box | BIOQIC | Simulation | ✓ | Low | Primary |
| 2a | FEM Abdomen | BIOQIC | Simulation | ✓ | Medium | Primary |
| 2b | Agar Phantom | ScienceDB | Real MRI | ✓ | Medium | Validation |
| 3 | Liver | ScienceDB | Real MRI | ✓ | High | Primary |
| 3* | Liver (backup) | BIOQIC | Real MRI | ? | High | Secondary |

---

## Implementation Sequence

### Week 1-2: Part 1 - Data Pipeline
- [ ] Download all relevant datasets (ScienceDB liver/phantom, BIOQIC box/abdomen)
- [ ] Build data loaders for `.mat` files
- [ ] Implement preprocessing (normalization, masking, coordinate grid generation)
- [ ] Visualize displacement fields and ground truth stiffness
- [ ] Generate collocation point samplers

### Week 3: Part 2 - Physics Module
- [ ] Implement automatic differentiation for ∇u, ∇²u
- [ ] Code homogeneous Helmholtz residual calculator
- [ ] Code heterogeneous Helmholtz residual (with ∇μ terms)
- [ ] Validate derivatives with finite difference checks
- [ ] Implement boundary condition functions

### Week 4-5: Part 3 - Phase 1 Implementation
- [ ] Build basic ELM network (single hidden layer)
- [ ] Train on BIOQIC Box data (forward problem)
- [ ] Validate displacement field fitting
- [ ] Check PDE residual convergence
- [ ] Document baseline results

### Week 6-8: Part 3 - Phase 2 Implementation
- [ ] Build dual network architecture (u-network + μ-network)
- [ ] Implement iterative training loop
- [ ] Add curriculum learning scheduler
- [ ] Train on BIOQIC Abdomen simulation
- [ ] Validate on ScienceDB Phantom
- [ ] Compare predicted vs ground truth elastograms

### Week 9-11: Part 3 - Phase 3 Implementation
- [ ] Integrate multi-frequency loss
- [ ] Add noise robustness mechanisms
- [ ] Train on ScienceDB Real Liver data
- [ ] Hyperparameter tuning (regularization, network size, iterations)
- [ ] Cross-validate with BIOQIC Liver (if available)
- [ ] Generate clinical elastogram visualizations

### Week 12: Analysis and Documentation
- [ ] Benchmark against baseline methods (direct inversion, MDEV)
- [ ] Quantitative comparison: MAE, SSIM, computation time
- [ ] Prepare figures and results for publication
- [ ] Code cleanup and documentation
- [ ] Write methods section for paper

---

## Key Technical Specifications

### Network Hyperparameters (Starting Point)
- **Hidden neurons:** 500-1000 (tune based on grid size)
- **Activation:** Sine or Tanh (smooth derivatives for physics)
- **Input normalization:** Scale coordinates to [-1, 1]
- **Output scaling:** Match physical units (Pa for μ, radians for u)

### Loss Function Weights (Initial)
- **Phase 1:** λ_data = 1.0, λ_physics = 0.1
- **Phase 2:** λ_data = 0.5, λ_physics = 0.5, λ_reg = 0.1
- **Phase 3:** λ_data = 1.0, λ_physics = 0.3, λ_reg = 0.01 (real data prioritizes measurements)

### Convergence Criteria
- Max iterations: 500-1000
- Stop if |μ_change| < 1e-4 for 50 consecutive iterations
- Or PDE residual < 1e-3

---

## Expected Challenges and Mitigations

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| Noisy displacement data | Increase hidden layer size, add Tikhonov regularization |
| Sharp stiffness boundaries | Use adaptive collocation point sampling near edges |
| Non-convex optimization | Multiple random initializations, ensemble predictions |
| Computational cost | Start with downsampled grids (32³), scale up progressively |
| Overfitting to noise | Early stopping, validation set monitoring |

---

## Success Criteria (Final)

### Quantitative:
- MAE < 10% of mean stiffness value
- SSIM > 0.85 vs ground truth
- Training time < 5 minutes per case (vs hours for PINN)

### Qualitative:
- Smooth elastograms without checkerboard artifacts
- Clear delineation of stiff/soft regions
- Physically plausible values (liver: 2-4 kPa healthy tissue)

### Scientific Contribution:
- First published PIELM-based MRE inversion
- 10-100x speedup vs PINN methods
- Open-source code for reproducibility

---

## References and Resources

### Datasets:
- **ScienceDB:** https://www.scidb.cn/en/detail?dataSetId=a68111835ceb4750b4d60abae4b962d9
  - Contact: fengyuan@sjtu.edu.cn
  - Citation: Yuan Feng et al. (2025). DOI:10.57760/sciencedb.22378
  
- **BIOQIC:** https://bioqic-apps.charite.de/downloads
  - Contact: bioqic-apps@charite.de
  - Reference: Streitberger et al. (2014) PLOS ONE 9(10): e110588

### Key Papers:
- PINN-MRE Liver Inversion (NIH 2023)
- Iterative PIELM (Blue-Giant et al.)
- Curriculum-Driven PIELM
- BIOQIC MDEV Direct Inversion Method

### Code Repository:
- Blue-Giant/PIELM_Numpy (GitHub)

---

## Next Immediate Steps
1. Download and inspect BIOQIC `four_target_phantom.mat`
2. Download ScienceDB liver dataset (`PhaseRaw_liver.mat`, `U_liver.mat`, `G_liver.mat`)
3. Set up Python environment with required packages (scipy, numpy, matplotlib)
4. Begin Part 1: Data loading and visualization notebook

---

**Document Version:** 1.0  
**Last Updated:** November 19, 2025  
**Status:** Ready for Implementation
