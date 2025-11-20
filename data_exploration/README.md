# Data Exploration - BIOQIC Phantom

This folder contains comprehensive data exploration and visualization scripts for the BIOQIC Phase 1 Box Phantom dataset.

## Purpose

Before attempting to solve the inverse MRE problem, we need to thoroughly understand:
- The geometry and spatial structure
- Displacement field characteristics (magnitude, phase, components)
- Ground truth stiffness distribution
- Wave propagation patterns
- Spatial correlations and variations

## Files

- `explore_bioqic_data.py` - Main exploration script
- `outputs/` - Generated visualizations and reports

## Usage

```bash
cd data_exploration
python explore_bioqic_data.py
```

This will generate:
1. **data_geometry.png** - 3D point cloud and 2D projections
2. **data_displacement_detailed.png** - Component-wise displacement analysis
3. **data_displacement_total.png** - Total displacement magnitude
4. **data_stiffness.png** - Ground truth stiffness distribution
5. **data_wave_phase.png** - Phase patterns for wave propagation
6. **data_spatial_variations.png** - Spatial correlation plots
7. **data_summary_report.txt** - Comprehensive numerical summary

## Key Insights

The exploration will help answer:
- What is the physical domain size?
- What are typical displacement magnitudes?
- Where are the stiffness contrasts?
- Which displacement component is dominant?
- Where should boundary conditions be applied?
- Is the data noisy or clean?
- What is the loss tangent (viscoelasticity)?

## Next Steps

After reviewing the exploration results:
1. Decide on displacement representation (magnitude vs components vs complex)
2. Identify proper boundary conditions from excitation patterns
3. Choose appropriate subsampling strategy
4. Determine physics parameter scaling
5. Design inversion experiments based on data characteristics
