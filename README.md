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
