# BIOQIC FEM Simulation Phantom Dataset: Comprehensive Documentation

## 1. Dataset Overview

The BIOQIC FEM (Finite Element Method) box simulation phantom dataset is a widely-used computational phantom for validating and benchmarking Magnetic Resonance Elastography (MRE) inverse problem algorithms. This dataset is publicly available on the BIOQIC platform (https://bioqic-apps.charite.de/downloads) and has been extensively employed in recent MRE research, including neural network-based inversion methods.

### Origin and Generation

- **Platform**: BIOQIC (Charité-Universitätsmedizin Berlin)
- **Source URL**: https://bioqic-apps.charite.de/downloads
- **Created by**: Cemre Ariyurek at the National Magnetic Resonance Research Center (UM-RAM), Bilkent University, Ankara, Turkey
- **FEM Software**: ABAQUS (Dassault Systèmes, France)
- **License/Availability**: Publicly available for download and use in research

---

## 2. Geometric Specifications

### Box Dimensions
- **Length (x-direction)**: 80 mm
- **Width (y-direction)**: 100 mm
- **Height (z-direction)**: 10 mm
- **Total domain**: 80 mm × 100 mm × 10 mm

### Mesh Properties
- **Mesh type**: Isotropic hexahedral elements
- **Element size**: 1 mm × 1 mm × 1 mm cubic voxels
- **Spatial resolution**: 1 mm isotropic in all directions
- **Total voxels**: ~80,000 voxels (80 × 100 × 10)

---

## 3. Phantom Structure: Four Cylindrical Inclusions

The phantom contains four cylindrical inclusions (stiff targets) of decreasing sizes embedded in a homogeneous background material.

### Inclusion Specifications

| Inclusion # | Radius (mm) | Diameter (mm) | Size Category |
|-------------|------------|--------------|---------------|
| 1           | 20         | 40           | Largest      |
| 2           | 10         | 20           | Large        |
| 3           | 4          | 8            | Medium       |
| 4           | 2          | 4            | Smallest     |

### Spatial Design Purpose
- **Resolution testing**: Inclusions of different sizes test the spatial resolution and contrast detectability of MRE reconstruction algorithms
- **Inclusion arrangement**: Four cylindrical targets arranged to simulate realistic heterogeneous tissue structure
- **Boundary effects**: Sharp material property boundaries at inclusion interfaces

---

## 4. Material Properties

### Constitutive Model: Voigt Viscoelastic Model

All materials follow a **linear isotropic incompressible viscoelastic** Voigt model.

### Background Material
- **Shear storage modulus (\(μ\))**: 3 kPa = 3,000 Pa
- **Shear viscosity (\(η\))**: 1 Pa·s
- **Complex shear modulus**: \(G^*(\omega) = 3000 + i\omega(1)\) Pa

### Inclusion Materials (Stiff Targets)
- **Shear storage modulus (\(μ\))**: 9 kPa = 9,000 Pa
- **Shear viscosity (\(η\))**: 1 Pa·s
- **Complex shear modulus**: \(G^*(\omega) = 9000 + i\omega(1)\) Pa
- **Contrast ratio**: 9 kPa / 3 kPa = 3.0× (three-fold increase)

### Material Assumptions
- **Isotropy**: Mechanical properties are identical in all directions (scalar modulus fields)
- **Incompressibility**: Poisson's ratio ≈ 0.5; bulk modulus >> shear modulus; divergence of displacement ≈ 0
- **Homogeneity within regions**: Properties are uniform within the background and uniform within each inclusion (piecewise constant)
- **Linearity**: Material response does not depend on strain amplitude (valid for small deformations typical in MRE)

### Stiffness Field Representation
- **Dimensionality**: 3D scalar fields (\(μ\) and \(η\)) defined on a regular 1 mm³ voxel grid
- **Spatial structure**: 
  - Uniform background values (3 kPa, 1 Pa·s)
  - Higher values inside cylindrical inclusion regions (9 kPa, 1 Pa·s)
  - Sharp step changes at material boundaries
  - No gradual transitions or smoothing

---

## 5. Excitation Frequencies

### Frequency Range
- **Minimum**: 50 Hz
- **Maximum**: 100 Hz
- **Increment**: 10 Hz
- **Available frequencies**: 50, 60, 70, 80, 90, 100 Hz

### Multifrequency Data
- Data are provided at each frequency independently
- Enables multifrequency MRE analysis and frequency-dependent property estimation
- Allows testing of algorithms that combine data across multiple frequencies

---

## 6. Boundary Conditions

### Excitation Boundary
- **Location**: Top xz-plane (y = 100 mm)
- **Excitation type**: Surface traction (stress boundary condition)
- **Vibration mode**: Steady-state sinusoidal vibration at prescribed frequency
- **Direction**: Typically shear traction normal to surface

### Absorbing Boundaries
- **Location**: All other sides (x = 0, x = 80 mm, z = 0, z = 10 mm, and potentially y = 0)
- **Purpose**: Prevent wave reflections and simulate open/infinite medium conditions
- **Implementation**: Absorbent boundary conditions (radiation boundary conditions or similar)
- **Effect**: Waves propagate away from the domain without bouncing back

---

## 7. Governing Equations

### 3D Full Governing Equation

The BIOQIC FEM box simulation solves the **isotropic incompressible linear viscoelastic wave equation** in frequency domain:

\[
-\omega^2 \rho \mathbf{U}(\mathbf{x}, \omega) = 
\nabla \cdot \left[ (\mu(\mathbf{x}) + i \omega \eta(\mathbf{x})) \left( \nabla \mathbf{U} + (\nabla \mathbf{U})^T \right) \right] + \mathbf{F}_{\text{ext}}(\mathbf{x}, \omega)
\]

Where:
- \(\mathbf{x} = (x, y, z)\): spatial coordinates (mm)
- \(\omega = 2\pi f\): angular frequency (rad/s)
- \(\mathbf{U}(\mathbf{x}, \omega) = (U_x, U_y, U_z)\): complex-valued 3D displacement vector (mm)
- \(\rho\): mass density (kg/m³), typically assumed constant ~1000 kg/m³ for tissue
- \(\mu(\mathbf{x})\): spatial map of shear storage modulus (Pa)
- \(\eta(\mathbf{x})\): spatial map of shear viscosity (Pa·s)
- \(\mathbf{F}_{\text{ext}}(\mathbf{x}, \omega)\): externally applied force per unit volume (Pa or N/m³)
- \(i = \sqrt{-1}\): imaginary unit
- \(\nabla\): gradient operator
- \(\nabla \cdot\): divergence operator

#### Alternative Time-Domain Form (Optional Reference)
For context, the time-domain equation is:

\[
\rho \frac{\partial^2 \mathbf{u}}{\partial t^2} =
\nabla \cdot \left[ \mu(\mathbf{x}) \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) + \eta(\mathbf{x}) \frac{\partial}{\partial t} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right] + \mathbf{f}_{\text{ext}}(\mathbf{x}, t)
\]

where \(\mathbf{u}(\mathbf{x}, t)\) is the real displacement field in time, and assuming harmonic excitation \(\mathbf{u}(\mathbf{x}, t) = \Re[\mathbf{U}(\mathbf{x}) e^{i \omega t}]\).

### 2D Simplified Governing Equation

When using a 2D slice from the 3D data (e.g., a horizontal slice at constant z):

\[
-\omega^2 \rho \mathbf{U}_{2D}(\mathbf{x}_{2D}, \omega) = 
\nabla_{2D} \cdot \left[ (\mu(\mathbf{x}_{2D}) + i \omega \eta(\mathbf{x}_{2D})) \nabla_{2D} \mathbf{U}_{2D} \right] + \mathbf{F}_{\text{ext}}^{2D}(\mathbf{x}_{2D}, \omega)
\]

Where:
- \(\mathbf{x}_{2D} = (x, y)\): 2D spatial coordinates
- \(\mathbf{U}_{2D}(\mathbf{x}_{2D}, \omega) = (U_x, U_y)\): 2D in-plane displacement vector (U_z = 0 or ignored)
- \(\nabla_{2D}, \nabla_{2D} \cdot\): 2D gradient and divergence operators
- All material properties and external forces are projected onto the 2D slice

---

## 8. Assumptions and Their Implications

### Physical Assumptions

| Assumption | Statement | Implication | Validity Range |
|-----------|-----------|------------|-----------------|
| **Linear Viscoelasticity** | Material stress-strain response is linear; no strain amplitude dependence | Simplifies to frequency-dependent complex modulus; enables superposition | Small deformations (strain < 1%) |
| **Isotropy** | Mechanical properties identical in all spatial directions | Scalar modulus fields instead of tensors; simplifies computation | Most soft tissues ~isotropic |
| **Incompressibility** | Bulk modulus >> Shear modulus; \(\nu \approx 0.5\) | Divergence of displacement ≈ 0; only shear modes significant | Most soft tissues, ~incompressible |
| **Homogeneity (piecewise)** | Properties constant within each region (background or inclusion) | No property gradients within regions; sharp boundaries at interfaces | Realistic for tissue/inclusion transitions |
| **Voigt Viscoelasticity** | Elastic and viscous effects act in parallel | Simple frequency dependence; captures basic wave attenuation | Typical for soft tissues 10–1000 Hz |

### Computational Assumptions

| Assumption | Statement | Implication |
|-----------|-----------|------------|
| **Steady-State Excitation** | Sinusoidal vibration at fixed frequency; transients negligible | Solutions computed in frequency domain; reduced computation vs. time-stepping |
| **Frequency Domain** | Work with complex amplitudes and angular frequency | Fast computation; enables efficient PINN/neural network training on frequency data |
| **Absorbent Boundaries** | No wave reflection from outer domain boundaries | Simulates open/infinite medium; realistic MRE-like conditions |
| **Constant Mass Density** | \(\rho\) = 1000 kg/m³ (assumed uniform) | Simplifies PDE; typically valid since density variations < 5% in soft tissue |

---

## 9. Displacement Field Properties

### Dimensionality and Structure

- **Field type**: 3D complex-valued vector field \(\mathbf{U}(\mathbf{x}, \omega) = (U_x, U_y, U_z)\)
- **Domain**: Defined on every voxel in the 80 × 100 × 10 mm³ box
- **Data type**: Complex numbers (real amplitude + imaginary phase for each component)
- **Typical representation**: Magnitude \(|\mathbf{U}|\) and phase \(\angle \mathbf{U}\) are often stored/visualized

### Spatial Variability

- **Near excitation source** (top plane, y ≈ 100 mm): Highest displacement amplitudes
- **With propagation distance**: Amplitude attenuates due to viscoelastic damping (\(\eta\) term)
- **Near stiff inclusions**: Displacement field reflects and refracts at material boundaries (impedance mismatch)
- **Local amplitude modulation**: Increased gradients and phase shifts where inclusions are present
- **Wave pattern**: At each frequency, standing wave or quasi-steady-state patterns develop

### Typical Magnitude Ranges

- **Background region, near source**: 0.1–1.0 mm peak displacement
- **Background region, far from source**: 0.01–0.1 mm peak displacement
- **Near inclusion boundaries**: Localized enhancement or suppression depending on wave mode
- **Phase variation**: 0–2π radians across the domain

### Frequency Dependence

- **Higher frequencies** (80–100 Hz):
  - Shorter wavelength
  - More rapid spatial variation
  - Stronger attenuation over distance
  - More localized effects near inclusions
- **Lower frequencies** (50–60 Hz):
  - Longer wavelength
  - Smoother spatial patterns
  - Less attenuation
  - More global wave propagation

---

## 10. Stiffness Field Properties

### Storage Modulus (\(μ\)) Field

#### Dimensionality
- **Type**: 3D scalar field (real-valued)
- **Domain**: Defined at each voxel center (80 × 100 × 10 grid)
- **Unit**: Pascal (Pa) or kilopascal (kPa)

#### Spatial Variability
- **Piecewise constant structure**:
  - Background region: 3 kPa everywhere except inside inclusions
  - Inclusion regions: 9 kPa uniformly within each cylindrical target
  - Boundaries: Sharp step changes (discontinuities) at tissue/inclusion interfaces
- **No gradual transitions**: Property value changes instantaneously at boundaries
- **Contrast**: 9 kPa / 3 kPa = 3.0× stiffness ratio (high contrast for algorithm testing)

#### Spatial Distribution
- **Largest inclusion** (radius 20 mm): Occupies significant volume; tests large heterogeneity
- **Smallest inclusion** (radius 2 mm): Tests resolution limit and small target detectability
- **Arrangement**: Four targets provide multi-scale testing capability

### Shear Viscosity (\(η\)) Field

#### Dimensionality
- **Type**: 3D scalar field (real-valued)
- **Domain**: Defined at each voxel center (same grid as \(μ\))
- **Unit**: Pascal·second (Pa·s)

#### Spatial Variability
- **Uniform across domain**: η = 1 Pa·s everywhere (background and inclusions)
- **Frequency scaling**: Viscous effect scaled by frequency: \(i \omega \eta\)
- **At 50 Hz**: Viscous term magnitude ≈ 314 Pa (comparable to storage modulus)
- **At 100 Hz**: Viscous term magnitude ≈ 628 Pa (~6–7% of storage modulus magnitude)

### Complex Shear Modulus (\(G^* = \mu + i \omega \eta\))

#### Frequency Dependence

**At background (μ = 3 kPa):**
- f = 50 Hz: \(G^* = 3000 + i(314) = 3016 \angle 6.0°\) Pa
- f = 100 Hz: \(G^* = 3000 + i(628) = 3066 \angle 11.8°\) Pa

**At inclusion (μ = 9 kPa):**
- f = 50 Hz: \(G^* = 9000 + i(314) = 9005 \angle 2.0°\) Pa
- f = 100 Hz: \(G^* = 9000 + i(628) = 9022 \angle 4.0°\) Pa

#### Phase Lag
- **Low frequencies**: Phase lag (loss angle) is small; material behaves nearly elastically
- **Higher frequencies**: Phase lag increases; viscous dissipation becomes more significant

---

## 11. Data Organization and Format

### Available Data Files

From the BIOQIC downloads platform, typical files include:

1. **Displacement fields** for each frequency
   - Format: 3D arrays (complex-valued or stored as magnitude + phase)
   - Files per frequency: Usually separate .npy, .mat, or .h5 format
   - Naming convention: e.g., `displacement_50Hz.npy`, `displacement_100Hz.npy`

2. **Stiffness maps** (ground truth)
   - Format: 3D array (integer or float labels)
   - Single file representing all inclusions and background
   - Naming: e.g., `stiffness_map.npy` or `ground_truth.mat`

3. **Metadata/Documentation**
   - Frequency list, grid dimensions, material parameters
   - Units and scaling factors
   - Boundary condition descriptions

### Data Dimensionality Summary

| Data | Dimensions | Shape | Data Type | Units |
|------|-----------|-------|-----------|-------|
| Displacement (3D) | 80 × 100 × 10 × 3 | (nx, ny, nz, 3 components) | Complex float | mm |
| Displacement (2D slice) | 80 × 100 × 3 | (nx, ny, 3 components) | Complex float | mm |
| Stiffness map | 80 × 100 × 10 | (nx, ny, nz) | Float or int | Pa |
| Viscosity map | 80 × 100 × 10 | (nx, ny, nz) | Float | Pa·s |

---

## 12. Use Cases and Applications

### 2D Usage (Slice-Based Methods)

**When to use:**
- Quick algorithm prototyping and testing
- Reduced computational cost (ideal for GPU training)
- Compatibility with 2D neural network architectures
- Fast iteration during method development

**How to use:**
- Extract a single z-slice (e.g., z = 5 mm, the middle slice)
- Use corresponding 2D displacement fields and 2D stiffness map
- Solve reduced 2D PDE for forward/inverse problems

**Trade-offs:**
- Loses out-of-plane wave effects
- May underestimate or misrepresent boundary effects
- Results may not generalize perfectly to true 3D cases

### 3D Usage (Full Volume Methods)

**When to use:**
- Realistic MRE simulation and algorithm validation
- Publication-quality results and high-fidelity benchmarking
- Testing algorithms designed for true 3D geometry
- Capturing wave mode coupling and edge effects

**How to use:**
- Use full 3D displacement fields and 3D stiffness maps
- Solve complete 3D frequency-domain PDE
- Apply PINN/neural network training on full volumetric data

**Advantages:**
- Captures complete physics
- Realistic boundary and mode effects
- Better generalization to experimental MRE

---

## 13. Applications in Recent Research

### Neural Network-Based Inversion

- **MICCAI 2023**: Physics-Informed Neural Networks (PINNs) for tissue elasticity reconstruction validated on BIOQIC phantom
- **MICCAI 2025**: Multifrequency neural network-based wave inversion (MF-ElastoNet) tested on BIOQIC FEM data

### Physics-Informed ELM (PI-ELM)

- Dataset suitable for PI-ELM variant development and testing
- Known ground truth enables quantitative validation
- Multiple frequencies support multifrequency PI-ELM approaches

### Algorithm Comparison

- **HMDI (Heterogeneous Multifrequency Direct Inversion, 2018)**: Validated against BIOQIC phantom
- **Local Frequency Estimation (LFE)**
- **MDEV (Multifrequency Dual Elasto-Visco) inversion**
- **Wavenumber-based k-MDEV**

---

## 14. Summary of Key Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| **Box dimensions** | 80 × 100 × 10 | mm |
| **Voxel size** | 1 × 1 × 1 | mm³ |
| **Total voxels** | ~80,000 | — |
| **Background shear modulus** | 3 | kPa |
| **Inclusion shear modulus** | 9 | kPa |
| **Modulus ratio (inclusion:background)** | 3:1 | — |
| **Shear viscosity (all regions)** | 1 | Pa·s |
| **Number of inclusions** | 4 | — |
| **Inclusion radii** | 20, 10, 4, 2 | mm |
| **Frequency range** | 50–100 | Hz |
| **Frequency resolution** | 10 | Hz |
| **Number of frequencies** | 6 | — |
| **Mesh type** | Hexahedral isotropic | — |
| **Boundary conditions** | Absorbent on sides, traction on top | — |
| **Viscoelastic model** | Voigt (linear) | — |
| **Material assumptions** | Isotropic, incompressible | — |

---

## 15. Accessing and Using the Dataset

### Download Location
- **URL**: https://bioqic-apps.charite.de/downloads

### Processing Tools
The BIOQIC platform provides built-in inversion methods:
- Local Frequency Estimation (LFE)
- MDEV inversion
- Neural network-based methods
- Custom algorithm integration

### Recommended Preprocessing
1. Load displacement fields and stiffness map
2. Verify grid alignment and voxel spacing (1 mm)
3. Check frequency assignment for each displacement file
4. Normalize or scale if using with neural networks
5. Select 2D slice or use full 3D volume depending on algorithm

---

## 16. References and Further Information

- **BIOQIC Main Portal**: https://bioqic.de/
- **BIOQIC Publications**: https://bioqic.de/publications/
- **BIOQIC Cloud Tools**: https://bioqic.de/bioqic-cloud/

### Key Related Literature

- Barnhill et al., 2018: "Heterogeneous Multifrequency Direct Inversion (HMDI) for Magnetic Resonance Elastography" — foundational validation study
- MICCAI 2023, 2025: Recent neural network applications validated on BIOQIC phantom
- Nature Scientific Data (2024–2025): MRE datasets including phantom, liver, and brain

---

## Appendix: Mathematical Notation Reference

| Symbol | Meaning | Unit |
|--------|---------|------|
| \(\mathbf{U}\) or \(\mathbf{u}\) | Displacement vector (complex in FD, real in TD) | mm |
| \(U_x, U_y, U_z\) | x, y, z components of displacement | mm |
| \(\omega\) | Angular frequency = \(2\pi f\) | rad/s |
| \(f\) | Frequency | Hz |
| \(\rho\) | Mass density | kg/m³ |
| \(\mu\) | Shear storage modulus (elastic part) | Pa |
| \(\eta\) | Shear viscosity (viscous damping) | Pa·s |
| \(G^*(\omega)\) | Complex shear modulus = \(\mu + i\omega\eta\) | Pa |
| \(\mathbf{F}_{\text{ext}}\) | External force per unit volume | Pa or N/m³ |
| \(\nabla\) | Gradient operator | 1/mm |
| \(\nabla \cdot\) | Divergence operator | 1/mm |
| \(\Re[]\) | Real part | — |
| \(i\) | Imaginary unit, \(\sqrt{-1}\) | — |

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Compiled for**: MRE Inverse Problem Research with Physics-Informed Neural Networks