# PIELM-MRE: Physics-Informed Extreme Learning Machine for MRE Inverse Problem

Implementation of Physics-Informed Extreme Learning Machine (PIELM) with Iterative Learning and Curriculum Learning for solving the Magnetic Resonance Elastography (MRE) inverse problem in liver tissue.

## Project Overview

This project implements a novel approach to estimate tissue stiffness (complex shear modulus) from MRE displacement measurements by solving the heterogeneous Helmholtz equation using PIELM - a faster alternative to Physics-Informed Neural Networks (PINNs).

### Key Features
- **Iterative PIELM Architecture**: Dual network system for coupled displacement-modulus estimation
- **Curriculum Learning**: Progressive complexity from homogeneous to heterogeneous physics
- **Multi-Phase Implementation**: Structured progression from simulated to real clinical data
- **10-100x Faster**: Compared to traditional PINN methods

## Project Structure

```
MRE_Visualisations/
├── data/
│   ├── raw/                    # Raw datasets (not tracked in git)
│   │   ├── bioqic/            # BIOQIC FEM simulations
│   │   └── scidb/             # ScienceDB phantom and liver data
│   └── processed/             # Preprocessed data (generated)
│
├── Draft 1/                   # Reference implementations
├── outputs/                   # Results and visualizations
│
├── PIELM-MRE-README.md       # Detailed technical documentation
├── plan.md                    # Implementation roadmap
├── PIELM_solver_v2.ipynb     # Reference solver notebook
└── requirements.txt           # Python dependencies
```

## Implementation Phases

### Phase 1: Sanity Check (Homogeneous/Forward Problem)
- **Dataset**: BIOQIC FEM Box Simulation
- **Goal**: Validate ELM can fit displacement data and satisfy basic PDE
- **Physics**: Homogeneous Helmholtz equation

### Phase 2: Coupled Solver (Heterogeneous/Inverse Problem)
- **Dataset**: BIOQIC FEM Abdomen Simulation + ScienceDB Phantom
- **Goal**: Solve for spatial stiffness distribution
- **Physics**: Full heterogeneous Helmholtz equation with curriculum learning

### Phase 3: Robust Solver (Real Clinical Data)
- **Dataset**: ScienceDB Real Liver MRE
- **Goal**: Handle noise, irregular boundaries, multi-frequency data
- **Enhancements**: Noise robustness, multi-frequency joint inversion

## Datasets

### Required Data Sources

**BIOQIC** (https://bioqic-apps.charite.de/downloads)
- FEM Box Simulation (Phase 1)
- FEM Abdomen Simulation (Phase 2)

**ScienceDB** (https://www.scidb.cn/en/detail?dataSetId=a68111835ceb4750b4d60abae4b962d9)
- Agar Phantom (Phase 2 validation)
- Real Liver MRE (Phase 3)

*Note: Raw data files (.mat) are not included in this repository. Download them from the sources above and place in `data/raw/` following the structure in `plan.md`.*

## Installation

```bash
# Clone the repository
git clone https://github.com/yeshu183/LLM-PIELM.git
cd LLM-PIELM

# Install dependencies
pip install -r requirements.txt

# Download datasets (see plan.md for details)
```

## Usage

```python
# TODO: Add usage examples once implementation is complete
```

## Methodology

### Iterative PIELM Algorithm
1. **Network A (Displacement)**: Approximates u(x,y,z) from coordinates
2. **Network B (Modulus)**: Approximates μ(x,y,z) from coordinates
3. **Iterative Training Loop**:
   - Fix μ, train u to match data and satisfy PDE
   - Fix u, train μ to minimize PDE residual
   - Repeat until convergence

### Curriculum Learning Schedule
- **Stage 1**: High regularization (near-homogeneous approximation)
- **Stage 2**: Reduced regularization (allow moderate heterogeneity)
- **Stage 3**: Full heterogeneity (capture sharp boundaries)

## Expected Results

### Quantitative Metrics
- MAE < 10% of mean stiffness value
- SSIM > 0.85 vs ground truth
- Training time < 5 minutes per case (vs hours for PINN)

### Scientific Contribution
- First published PIELM-based MRE inversion
- 10-100x speedup vs PINN methods
- Open-source reproducible implementation

## Timeline

- **Week 1-2**: Data loading and preprocessing
- **Week 3**: Physics module implementation
- **Week 4-5**: Phase 1 implementation
- **Week 6-8**: Phase 2 implementation
- **Week 9-11**: Phase 3 implementation
- **Week 12**: Analysis and documentation

## References

### Key Papers
- PINN-MRE Liver Inversion (NIH 2023)
- Iterative PIELM (Blue-Giant et al.)
- Curriculum-Driven PIELM
- BIOQIC MDEV Method: Streitberger et al. (2014) PLOS ONE 9(10): e110588

### Datasets
- Yuan Feng et al. (2025). DOI:10.57760/sciencedb.22378
- BIOQIC-Charite: bioqic-apps@charite.de

## License

[Add your license here]

## Contact

- **Author**: Yeshwanth Kesav
- **Repository**: https://github.com/yeshu183/LLM-PIELM

## Acknowledgments

- BIOQIC-Charite for FEM simulation data
- ScienceDB/Yuan Feng research group for MRE datasets
- Blue-Giant for PIELM reference implementations

---

**Status**: 🚧 In Development - Phase 1 Data Preprocessing

**Last Updated**: November 19, 2025
