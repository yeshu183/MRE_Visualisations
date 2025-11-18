# PIELM Solver — Full Config & Code Guide (Unified **+nu** Convention)

This guide explains **what the solver solves**, **how to configure it**, and **how the code works** so teammates can build a text→config→solve pipeline confidently.


---

## 1) Governing PDEs (normalized forms)

The solver assembles a single linear system from collocation rows that encode one of these canonical forms:

### 1.1 Steady problems

* **1D Advection**: $a(x)\,u_x = R(x)$
* **1D Diffusion/Poisson (special branch)**: $u_{xx} = R(x)$
  (This branch **ignores** `nu` by design; use `advdiff` with `a=0` if you need a scaled Laplacian in 1D steady.)
* **1D Advection–Diffusion**: $a(x)\,u_x + \nu\,u_{xx} = R(x)$
* **2D Advection–Diffusion**: $a(x,y)\,u_x + b(x,y)\,u_y + \nu\,(u_{xx}+u_{yy}) = R(x,y)$

### 1.2 Unsteady problems

* **1D Advection**: $u_t + a(x)\,u_x = R(x,t)$
* **1D Diffusion (heat)**: $u_t + \nu\,u_{xx} = R(x,t)$
* **1D Advection–Diffusion**: $u_t + a(x)\,u_x + \nu\,u_{xx} = R(x,t)$
* **2D Advection–Diffusion**: $u_t + a(x,y)\,u_x + b(x,y)\,u_y + \nu\,(u_{xx}+u_{yy}) = R(x,y,t)$

> **Mapping to classical forms:** If the physics is $u_t - \nu_0\,\Delta u = R$ (with a minus), pass `nu = -nu0`. Likewise for 1D steady $a u_x - \nu_0 u_{xx} = R$, pass `nu = -nu0`.

---

## 2) Config dictionary — all parameters and allowed values

All keys are optional unless marked **(req)**.

### 2.1 Top-level keys

* **`dim`** *(req)*: `1` or `2`
  Spatial dimension.

* **`time`** *(req)*: `False` (steady) or `True` (unsteady)
  Chooses whether $u_t$ terms appear and whether initial conditions are needed.

* **`operator`** *(req)*: `{'type': 'adv' | 'diff' | 'poisson' | 'advdiff'}`
  Aliases allowed: `'advection'`, `'diffusion'`, `'advection-diffusion'`, `'adv-diff'`.
  • *Special case*: for **1D steady** with `type in {'diff','poisson'}`, the solver enforces **`u_xx = R`** and **ignores** `nu`.

* **`coeffs`** *(req for the terms you use)*:

  * `a`: advection coefficient. Scalar or callable.

    * 1D steady/unsteady: `a(x)` or float.
    * 2D steady/unsteady: `a(x,y)` or float.
  * `b`: y-advection coefficient (2D only). Scalar or callable `b(x,y)`.
  * `nu`: diffusion coefficient (float). Use **positive** for `+nu Δu`, **negative** for `- |nu| Δu`.
    *Legacy:* `'gamma'` is accepted and mapped internally to `nu`.

* **`forcing`** *(req)*: Right-hand side $R$. A float or callable of the correct arity:

  * 1D steady → `R(x)`
  * 1D unsteady → `R(x,t)`
  * 2D steady → `R(x,y)`
  * 2D unsteady → `R(x,y,t)`

* **`domain`** *(req)*:

  * 1D: `{'type':'interval','x': (xL,xR)}`
  * 2D rectangle: `{'type':'rect','x': (xL,xR), 'y': (yL,yR)}`
  * 2D polygon: `{'type':'polygon','verts': array_like(Nv,2)}`

* **`bc`** *(req)*: Boundary conditions.

  * **1D**:
    `{'type':'dirichlet','gL': ..., 'gR': ...}` or
    `{'type':'neumann','gL': ..., 'gR': ...}` (values are target $u_x$ at ends) or
    `{'type':'periodic'}`.
  * **2D**:
    `{'type':'dirichlet','g': value | g(x,y) | g(x,y,t)}` (Neumann/Robin can be added similarly).

* **`ic`** *(req if `time=True`)*: Initial condition at `t=0`.

  * 1D: `{'F': F(x)}`
  * 2D: `{'F': F(x,y)}`

* **`tmax`** *(unsteady)*: Float. Upper bound for time sampling of interior/boundary points.

### 2.2 Numerical & modeling keys

* **`collocation`**: `{'N_f': int, 'N_bc': int, 'N_ic': int}`
  Counts of interior PDE rows, boundary rows, and initial rows (IC used only if `time=True`).

* **`model`**: `{'N_star': int, 'seed': int, 'ridge': float, 'rcond': float}`

  * `N_star`: number of random features (hidden units).
  * `seed`: RNG seed for features and sampling.
  * `ridge`: ridge parameter $\lambda$ in `(HᵀH + λI)c = HᵀK`. If 0, uses pseudoinverse.
  * `rcond`: cutoff passed to `pinv`.

* **`weights`** *(optional)*: `{'pde': float, 'bc': float, 'ic': float}`
  Row weights in the stacked system to emphasize PDE vs. BC/IC.

* **`plot`** *(optional)*: e.g. `{'enabled': True, 'n_plot': 1000, 'times': [...], 'surface3d': False}`.

* **`exact`** *(optional)*: Exact solution callable used only for plotting/errors.

---

## 3) How parameters determine the governing equation

Given `dim`, `time`, and `operator.type`, the solver chooses a target PDE and inserts your coefficients/functions:

* **Advection only (`adv`)**

  * 1D steady: $a(x)\,u_x = R(x)$
  * 1D unsteady: $u_t + a(x)\,u_x = R(x,t)$
  * 2D steady: $a\,u_x + b\,u_y = R(x,y)$
  * 2D unsteady: $u_t + a\,u_x + b\,u_y = R(x,y,t)$

* **Diffusion/Poisson (`diff` or `poisson`)**

  * 1D steady: **always** $u_{xx} = R(x)$, `nu` ignored.
    (*If you need scaled Laplacian in 1D steady, pick `advdiff` with `a=0`, desired `nu`.*)
  * 2D steady: $\nu\,(u_{xx}+u_{yy}) = R(x,y)$
  * 1D unsteady: $u_t + \nu\,u_{xx} = R(x,t)$
  * 2D unsteady: $u_t + \nu\,(u_{xx}+u_{yy}) = R(x,y,t)$

* **Advection–Diffusion (`advdiff`)**

  * 1D steady: $a(x)\,u_x + \nu\,u_{xx} = R(x)$
  * 1D unsteady: $u_t + a(x)\,u_x + \nu\,u_{xx} = R(x,t)$
  * 2D steady: $a\,u_x + b\,u_y + \nu\,(u_{xx}+u_{yy}) = R(x,y)$
  * 2D unsteady: $u_t + a\,u_x + b\,u_y + \nu\,(u_{xx}+u_{yy}) = R(x,y,t)$

**Effect of individual parameters**

* `a`, `b` (scalars or functions): scale the advection terms. Setting them to `0` removes advection. Non-constant functions permit spatially varying transport.
* `nu` (float): scales the Laplacian.
  • `nu > 0` → $+\nu\,\Delta u$
  • `nu < 0` → $-|\nu|\,\Delta u$ (classical heat/viscous form)
  • **Ignored** only in the 1D-steady `diff/poisson` branch.
* `forcing` (`R`): the source term; can be zero or a function of space(/time).
* `bc`, `ic`: determine boundary/initial rows; strong enforcement can be increased via `N_bc` and/or `weights['bc']`.
* `collocation` sizes + `model.N_star`: control the size/conditioning of the linear system (accuracy vs runtime).
* `ridge`: helps stabilize the solve when features are large or rows are nearly collinear.

---

## 4) Code walkthrough (what happens under the hood)

Below is a high‑level tour of the main blocks in `solve_pielm` and helpers.

### 4.1 Activations and exact derivatives

* `phi = tanh`, `phi_p`, `phi_pp` are closed‑form; we compute $u$, $u_x$, and $u_{xx}$ (and analogs for $y,t$) via the chain rule on the random features.

### 4.2 `_as_fun(v, nargs)`

Coerces a scalar or callable into a function of the requested arity so we can uniformly call `a(x)`, `a(x,y)`, `R(x,t)`, etc.

### 4.3 Parsing the problem

* Read `dim`, `time`, `operator.type`, `coeffs`, `forcing`, `domain`, `bc`, `ic`, and numeric options.
* **Unify diffusion key**: accept legacy `gamma` but map to `nu` internally.

### 4.4 Random feature layer (ELM)

* Build a design matrix using `W_in ∼ N(0,1)` and `b_in ∼ N(0,1)`.
* For an input point matrix `P` (columns in x\[,y\[,t]]), compute `Z = P W_inᵀ + b_inᵀ`, then the hidden features `phi(Z)` and its derivatives.

### 4.5 Collocation sampling

* **Interior**: Halton points inside the domain (1D, 2D rectangle, or filtered by polygon mask).
* **Boundary**: interval endpoints (1D); perimeter sampling for rectangles or polygon vertices/edges for polygons.
* **Initial**: a small tensor grid at `t=0` for unsteady cases.

### 4.6 PDE row assembly

For each interior point set we construct rows for the chosen PDE type using the chain rule:

* `phi_p(Z) * m.T` provides first derivatives ($u_x,u_y,u_t$).
* `phi_pp(Z) * (m.T**2)` and sums thereof provide second derivatives ($u_{xx},u_{yy}$).
* The solver **always** assembles **`+ nu * Laplacian`**. Passing `nu<0` flips the sign in physics space without changing code.
* **Special case**: 1D steady `diff/poisson` uses `phi_pp(Z)*(m.T**2)` alone ($u_{xx}$ only), ignoring `nu`.

### 4.7 Boundary/Initial rows

* **1D**: Dirichlet rows use `phi(Z_at_boundary)`; Neumann rows use `phi_p(Z)*m` (normal derivative). Periodic enforces equality between ends.
* **2D**: Dirichlet rows on boundary points (currently). Neumann/Robin can be added by projecting `grad u` onto boundary normals.

### 4.8 Linear solve

* Stack all rows: `H c = K`.
* If `ridge>0`: solve `(HᵀH + λI)c = HᵀK`. Else: `c = pinv(H)K`.
* Return `u_pred` that wraps the feature map and multiplies by `c`.

### 4.9 Plotting utilities

Convenience functions to compare to `exact` when available (1D/2D steady/unsteady, plus optional 3D surfaces).

---

## 5) Parameter‑by‑parameter impact on input & output

| Parameter       | Where used                           | Affects                            | Practical effect                                               |    |                                            |
| --------------- | ------------------------------------ | ---------------------------------- | -------------------------------------------------------------- | -- | ------------------------------------------ |
| `dim`           | sampling, feature dims, PDE assembly | selects 1D vs 2D rows              | Changes derivative structure & domain sampling.                |    |                                            |
| `time`          | sampling, PDE assembly, IC rows      | adds/removes `u_t` terms           | Requires `ic`, time sampling in `(0,tmax]`.                    |    |                                            |
| `operator.type` | PDE assembly                         | term structure                     | Chooses advection/diffusion/mix and special 1D Poisson branch. |    |                                            |
| `a`, `b`        | PDE assembly                         | advection strength                 | Directional transport; can be spatially varying.               |    |                                            |
| `nu`            | PDE assembly                         | Laplacian strength/sign            | `>0` for `+nu Δu`, `<0` for classical \`-                      | nu | Δu`. Ignored in 1D steady `diff/poisson\`. |
| `forcing`       | RHS                                  | source term                        | Drives solution away from homogeneous state.                   |    |                                            |
| `bc`            | boundary rows                        | solution values/fluxes at boundary | Stronger enforcement via `N_bc` or `weights['bc']`.            |    |                                            |
| `ic`            | initial rows (time=True)             | initial field                      | Sets `u(·,0)`; number of rows via `N_ic`.                      |    |                                            |
| `domain`        | sampling                             | geometry                           | Interval vs rectangle vs polygon affects collocation layout.   |    |                                            |
| `collocation`   | sampling                             | row counts                         | More rows → better coverage but bigger linear system.          |    |                                            |
| `model.N_star`  | features                             | expressivity/conditioning          | More features → richer basis; may require `ridge`.             |    |                                            |
| `model.ridge`   | solver                               | stability                          | Regularizes ill-conditioning.                                  |    |                                            |
| `weights`       | stacking                             | loss balance                       | Trade off PDE vs BC/IC residuals.                              |    |                                            |
| `plot`          | viz                                  | none                               | Controls plots and diagnostic surfaces.                        |    |                                            |

---

## 6) Recipes — from equation text to config ($+\nu$ normalized)

Below are quick templates you can paste into the parser’s output logic.

### 6.1 1D steady advection on \[0,1]

**PDE:** $u_x = R(x)$, $a=1$, Dirichlet at ends.

```python
problem = {
  'dim': 1, 'time': False,
  'operator': {'type': 'adv'},
  'coeffs': {'a': 1.0},
  'forcing': lambda x: R(x),
  'domain': {'type':'interval','x':(0.0,1.0)},
  'bc': {'type':'dirichlet','gL': gL, 'gR': gR},
}
```

### 6.2 1D steady diffusion/Poisson (special)

**PDE:** $u_{xx} = R(x)$ (ignores `nu`).

```python
problem = {
  'dim': 1, 'time': False,
  'operator': {'type': 'diff'},  # or 'poisson'
  'coeffs': {},                  # nu ignored here by design
  'forcing': lambda x: R(x),
  'domain': {'type':'interval','x':(0.0,1.0)},
  'bc': {'type':'dirichlet','gL': gL, 'gR': gR},
}
```

### 6.3 1D steady advection–diffusion with minus diffusion (classical)

**Physics:** $a u_x - \nu_0 u_{xx} = R$. **Config:** set `nu = -nu0` and use the $+\nu$ form.

```python
problem = {
  'dim': 1, 'time': False,
  'operator': {'type': 'advdiff'},
  'coeffs': {'a': a_fun, 'nu': -0.20},  # negative encodes the minus
  'forcing': lambda x: R(x),             # write R for the +nu form
}
```

### 6.4 1D unsteady diffusion (heat)

**Physics:** $u_t - \nu_0 u_{xx} = 0$. **Config:** `nu = -nu0`.

```python
problem = {
  'dim': 1, 'time': True, 'tmax': 1.0,
  'operator': {'type': 'diff'},
  'coeffs': {'a': 0.0, 'nu': -0.05},  # negative nu
  'forcing': lambda x,t: np.zeros_like(x),
  'domain': {'type':'interval','x':(0.0,1.0)},
  'bc': {'type':'dirichlet','gL': 0.0, 'gR': 0.0},
  'ic': {'F': lambda x: np.sin(np.pi*x)},
}
```

### 6.5 2D steady Poisson on \[0,1]^2

**PDE:** $u_{xx}+u_{yy} = R(x,y)$. Use `nu=1.0`.

```python
problem = {
  'dim': 2, 'time': False,
  'operator': {'type': 'diff'},
  'coeffs': {'a': 0.0, 'b': 0.0, 'nu': 1.0},
  'forcing': lambda x,y: R(x,y),
  'domain': {'type':'rect','x':(0.0,1.0),'y':(0.0,1.0)},
  'bc': {'type':'dirichlet','g': 0.0},
}
```

### 6.6 2D unsteady advection–diffusion with minus diffusion (classical)

**Physics:** $u_t + a u_x + b u_y - \nu_0 \Delta u = R$. **Config:** `nu = -nu0`.

```python
problem = {
  'dim': 2, 'time': True, 'tmax': 1.0,
  'operator': {'type': 'advdiff'},
  'coeffs': {'a': a_fun, 'b': b_fun, 'nu': -0.01},
  'forcing': lambda x,y,t: R(x,y,t),
  'domain': {'type':'rect','x':(0.0,1.0),'y':(0.0,1.0)},
  'bc': {'type':'dirichlet','g': lambda x,y,t: u_exact(x,y,t)},
  'ic': {'F': lambda x,y: u_exact(x,y,0.0)},
}
```

---

## 7) Notes, limitations, and extensions

* **Sign handling:** With the $+\nu$ convention, you never have to remember a built‑in minus; just set the sign in the config.
* **1D steady `diff/poisson`** ignores `nu`; if a scaled Laplacian is needed, switch to `advdiff` with `a=0`.
* **2D Neumann/Robin**: not yet implemented; would require normals along the boundary; structure mirrors 1D Neumann using `phi_p(Z)` projected onto normals.
* **Activation**: `tanh` chosen for smooth derivatives; you can swap by redefining `phi`, `phi_p`, `phi_pp`.
* **Conditioning**: large `N_star` or near‑singular rows may require a small `ridge` (e.g., `1e-8`).

---

## 8) Checklist for a text→config parser

1. Parse dimension and time dependence.
2. Normalize PDE into the $+\nu$ canonical form.
3. Extract or infer `operator.type`, `a`, `b`, `nu` (map any `gamma`→`nu`).
4. Extract `R` and domain.
5. Extract BCs; if unsteady, extract IC and `tmax`.
6. Choose sensible `collocation` and `model` defaults; bump `N_bc` or `weights['bc']` for hard Dirichlet.
7. (Optional) Attach an `exact` function for plots.

---

### Appendix: Quick mapping table

| Classical form in text               | Config value(s) to set                          |
| ------------------------------------ | ----------------------------------------------- |
| $u_t - \nu_0 \Delta u = R$           | `operator='diff' or 'advdiff'`, `nu = -nu0`     |
| $a u_x - \nu_0 u_{xx} = R$           | `operator='advdiff'`, `a=a(x[,y])`, `nu=-nu0`   |
| $u_{xx} = R$ (1D steady)             | `operator='diff'` (nu ignored)                  |
| $a u_x + b u_y + \nu_0 \Delta u = R$ | `operator='advdiff'`, `a,b` as given, `nu=+nu0` |
| Pure advection $u_t + a u_x = 0$     | `operator='adv'`, `a`, `nu=0`                   |

That’s it! With this guide and the $+\nu$ normalization, your teammates can read a PDE in prose, turn it into a clean config, and run `solve_pielm(problem)` without worrying about hidden sign flips in the solver.

