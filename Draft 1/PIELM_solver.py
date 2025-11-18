# PIELM_solver.py
# =============================================================
# Unified PIELM solver (normalized +nu convention)
# -------------------------------------------------------------
# - 1D/2D, steady/unsteady advection/diffusion/advection-diffusion
# - 1D steady 'diff/poisson' enforces u_xx = R (nu ignored)
# - For classical minus-Laplacian physics, pass a NEGATIVE nu
# - Returns (u_pred, info). If plot.enabled=True, shows figures.
# =============================================================

from typing import Callable, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Halton
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pi = np.pi

# ---------- activation & exact derivatives ----------
phi     = np.tanh
phi_p   = lambda z: 1.0 - np.tanh(z)**2
phi_pp  = lambda z: -2.0*np.tanh(z)*(1.0 - np.tanh(z)**2)

# ---------- helpers ----------
def _as_fun(v, nargs: int):
    """Return f(*xs) that yields array shape (N,1) from scalar/const/func."""
    if callable(v):
        return v
    else:
        if nargs == 1:
            return lambda x: np.full_like(x, float(v))
        if nargs == 2:
            return lambda x, y: np.full_like(x, float(v))
        if nargs == 3:
            return lambda x, y, t: np.full_like(x, float(v))

def _col(v, n_expected: int | None = None) -> np.ndarray:
    """
    Force v to a column array of shape (N,1).
    - If scalar and n_expected provided -> broadcast to (n_expected,1).
    - If (1,1) and n_expected provided -> repeat to (n_expected,1).
    - If (N, ) or (N,1) -> reshape to (N,1).
    - Else try flatten to (-1,1). If length mismatches and not broadcastable, raise.
    """
    a = np.asarray(v)
    if a.ndim == 0:
        if n_expected is None:
            return a.reshape(1,1)
        return np.full((n_expected, 1), float(a))
    if a.ndim == 1:
        a = a.reshape(-1,1)
    elif a.ndim == 2 and a.shape[1] != 1:
        a = a.reshape(-1,1)
    # Now a is (M,1)
    if n_expected is not None and a.shape[0] != n_expected:
        if a.shape[0] == 1:
            a = np.repeat(a, n_expected, axis=0)
        else:
            raise ValueError(f"RHS length {a.shape[0]} != expected {n_expected}")
    return a

# ---------- plotting blocks ----------
def _plot1d_steady(u_pred, u_exact, xL, xR, n, title=""):
    x = np.linspace(xL, xR, n).reshape(-1,1)
    up = u_pred(x).reshape(-1,1)
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(9,3.3), constrained_layout=True)
    if u_exact is not None:
        ue  = u_exact(x)
        ax0.plot(x, ue, 'b', lw=1.8, label='Exact')
        err = (up - ue)
    else:
        err = np.zeros_like(up)
    ax0.plot(x, up, 'r--', lw=1.8, label='PIELM')
    ax0.set_title(title or 'Solution')
    ax0.set_xlabel('x'); ax0.set_ylabel('u'); ax0.grid(alpha=.3); ax0.legend()
    ax1.plot(x, err, 'r', lw=1.5)
    ax1.set_title('Point-wise error')
    ax1.set_xlabel('x'); ax1.set_ylabel('error'); ax1.grid(alpha=.3)
    plt.show()

def _plot1d_unsteady(u_pred, u_exact, xL, xR, times, n, title=""):
    x = np.linspace(xL, xR, n).reshape(-1,1)
    fig, axs = plt.subplots(1, len(times), figsize=(10,3.3), constrained_layout=True)
    if len(times) == 1:
        axs = [axs]
    for j,tv in enumerate(times):
        tt = np.full_like(x, tv)
        up = u_pred(x, tt)
        if u_exact is not None:
            ue = u_exact(x, tt)
            axs[j].plot(x, ue, 'b', lw=1.8, label='Exact' if j==0 else "")
        axs[j].plot(x, up, 'r--', lw=1.8, label='PIELM' if j==0 else "")
        axs[j].set_title(f'{title}  t={tv}')
        axs[j].set_xlabel('x');
        if j==0: axs[j].set_ylabel('u')
        axs[j].grid(alpha=.3)
    if u_exact is not None: axs[0].legend(loc='upper center')
    plt.show()

def _plot2d_rect_steady(u_pred, u_exact, xL, xR, yL, yR, n, title=""):
    xs, ys = np.linspace(xL,xR,n), np.linspace(yL,yR,n)
    xx, yy = np.meshgrid(xs, ys)
    P = np.c_[xx.ravel(), yy.ravel()]
    up = u_pred(P).reshape(n,n)
    fig, axs = plt.subplots(1,2, figsize=(10,3.6), constrained_layout=True, sharex=True, sharey=True)
    m0 = axs[0].contourf(xx, yy, up, 50, cmap='jet'); axs[0].set_title(title+' – PIELM'); plt.colorbar(m0, ax=axs[0], fraction=.046)
    if u_exact is not None:
        ue = u_exact(xx, yy)
        err = up - ue
        m1 = axs[1].contourf(xx, yy, err, 50, cmap='jet'); axs[1].set_title('Point-wise error'); plt.colorbar(m1, ax=axs[1], fraction=.046)
    else:
        axs[1].axis('off')
    for a in axs: a.set_xlabel('x'); a.set_ylabel('y')
    plt.show()

def _plot2d_poly_steady(u_pred, u_exact, path: Path, n, title=""):
    xs = ys = np.linspace(0,1,n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.c_[xx.ravel(), yy.ravel()]
    mask = path.contains_points(pts)
    pts  = pts[mask]
    tri  = Triangulation(pts[:,0], pts[:,1])
    up   = u_pred(pts).ravel()
    fig,axs = plt.subplots(1,2, figsize=(10,3.6), constrained_layout=True)
    m0 = axs[0].tripcolor(tri, up, shading='gouraud', cmap='jet'); axs[0].set_title(title+' – PIELM'); plt.colorbar(m0, ax=axs[0], fraction=.046)
    if u_exact is not None:
        ue = u_exact(pts[:,0:1], pts[:,1:2]).ravel()
        err = up - ue
        m1 = axs[1].tripcolor(tri, err, shading='gouraud', cmap='jet'); axs[1].set_title('Point-wise error'); plt.colorbar(m1, ax=axs[1], fraction=.046)
    else:
        axs[1].axis('off')
    for a in axs: a.set_xticks([]); a.set_yticks([])
    plt.show()

def _plot2d_unsteady(u_pred, u_exact, xL, xR, yL, yR, times, n, title=""):
    xs, ys = np.linspace(xL,xR,n), np.linspace(yL,yR,n)
    xx, yy = np.meshgrid(xs, ys)
    fig, axs = plt.subplots(1, len(times), figsize=(13,3.6), constrained_layout=True, sharex=True, sharey=True)
    if len(times) == 1: axs = [axs]
    for ax, t in zip(axs, times):
        P  = np.c_[xx.ravel(), yy.ravel(), np.full((n*n,1), t)]
        up = u_pred(P).reshape(n,n)
        m  = ax.contourf(xx, yy, up, 50, cmap='jet'); ax.set(title=f'{title}  t={t}', xlabel='x', ylabel='y')
    fig.colorbar(m, ax=axs, shrink=0.85, label='u')
    plt.show()

def _plot2d_unsteady_error(u_pred, u_exact, xL, xR, yL, yR, times, n, title=""):
    if u_exact is None: return
    xs, ys = np.linspace(xL,xR,n), np.linspace(yL,yR,n)
    xx, yy = np.meshgrid(xs, ys)
    fig, axs = plt.subplots(1, len(times), figsize=(13,3.6), constrained_layout=True, sharex=True, sharey=True)
    if len(times) == 1: axs = [axs]
    for ax, t in zip(axs, times):
        P  = np.c_[xx.ravel(), yy.ravel(), np.full((n*n,1), t)]
        up = u_pred(P).reshape(n,n)
        ue = u_exact(xx, yy, t)
        err = up - ue
        m = ax.contourf(xx, yy, err, 50, cmap='jet'); ax.set(title=f'Error  t={t}', xlabel='x', ylabel='y')
    fig.colorbar(m, ax=axs, shrink=0.85, label='error')
    plt.show()

# 3D plotting helpers
def plot_3d_surface(u_pred, u_exact, xL, xR, yL, yR, n, title=""):
    xs, ys = np.linspace(xL, xR, n), np.linspace(yL, yR, n)
    xx, yy = np.meshgrid(xs, ys)
    P = np.c_[xx.ravel(), yy.ravel()]
    up = u_pred(P).reshape(n, n)
    ue = u_exact(xx, yy) if u_exact is not None else None
    err = up - ue if ue is not None else None

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    s1 = ax1.plot_surface(xx, yy, up, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_title(f'{title} - PIELM Solution'); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
    fig.colorbar(s1, ax=ax1, shrink=0.5, aspect=5)

    if ue is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        s2 = ax2.plot_surface(xx, yy, ue, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        ax2.set_title('Exact Solution'); ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
        fig.colorbar(s2, ax=ax2, shrink=0.5, aspect=5)

        ax3 = fig.add_subplot(133, projection='3d')
        s3 = ax3.plot_surface(xx, yy, err, cmap='coolwarm', alpha=0.8, linewidth=0, antialiased=True)
        ax3.set_title('Error'); ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('error')
        fig.colorbar(s3, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout(); plt.show()

def plot_3d_unsteady(u_pred, u_exact, xL, xR, yL, yR, times, n, title=""):
    xs, ys = np.linspace(xL, xR, n), np.linspace(yL, yR, n)
    xx, yy = np.meshgrid(xs, ys)
    for t in times:
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f'{title} - Time t={t}')
        P = np.c_[xx.ravel(), yy.ravel(), np.full((n*n,1), t)]
        up = u_pred(P).reshape(n, n)

        ax1 = fig.add_subplot(131, projection='3d')
        s1 = ax1.plot_surface(xx, yy, up, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        ax1.set_title('PIELM Solution'); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
        fig.colorbar(s1, ax=ax1, shrink=0.5, aspect=5)

        if u_exact is not None:
            ue = u_exact(xx, yy, t); err = up - ue
            ax2 = fig.add_subplot(132, projection='3d')
            s2  = ax2.plot_surface(xx, yy, ue, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
            ax2.set_title('Exact Solution'); ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
            fig.colorbar(s2, ax=ax2, shrink=0.5, aspect=5)

            ax3 = fig.add_subplot(133, projection='3d')
            s3  = ax3.plot_surface(xx, yy, err, cmap='coolwarm', alpha=0.8, linewidth=0, antialiased=True)
            ax3.set_title('Error'); ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('error')
            fig.colorbar(s3, ax=ax3, shrink=0.5, aspect=5)

        plt.tight_layout(); plt.show()

# ---------- main solver ----------
def solve_pielm(problem: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]:
    """
    Unified PIELM solver – builds H c = K for linear ADR-type PDEs.

    Canonical rows (normalized):
      Steady:   a u_x + b u_y +  nu (u_xx + u_yy) = R
      Unsteady: u_t + a u_x + b u_y +  nu (u_xx + u_yy) = R
    (If you need a minus Laplacian, pass a negative `nu` in `coeffs`.)
    1D steady 'diff/poisson' implements u_xx = R and IGNORES `nu`.
    """

    # --- parse config with defaults ---
    dim   = problem.get('dim', 1)
    timep = problem.get('time', False)
    op    = problem.get('operator', {'type':'advdiff'}); optype = op.get('type','advdiff').lower()

    coeffs = problem.get('coeffs', {})
    forcing= problem.get('forcing', 0.0)
    domain = problem.get('domain', {'type':'interval','x':(0.0,1.0)})

    bc     = problem.get('bc', {'type':'dirichlet'})
    ic     = problem.get('ic', None) if timep else None

    colloc = problem.get('collocation', {})
    N_f    = int(colloc.get('N_f', 400))
    N_bc   = int(colloc.get('N_bc', 40))
    N_ic   = int(colloc.get('N_ic', 40)) if timep else 0

    model  = problem.get('model', {})
    N_star = int(model.get('N_star', N_f + N_bc + N_ic))
    seed   = int(model.get('seed', 0))
    ridge  = float(model.get('ridge', 0.0))
    rcond  = float(model.get('rcond', 1e-8))

    exact  = problem.get('exact', None)

    plot   = problem.get('plot', {'enabled': True})
    do_plot= bool(plot.get('enabled', True))
    n_plot = int(plot.get('n_plot', 1000 if dim==1 else 160))
    times  = plot.get('times', [0.0, 0.25, 0.5]) if timep else []
    surf3d = bool(plot.get('surface3d', False))

    # ---- optional row weights ----
    weights = problem.get('weights', {})
    w_pde = float(weights.get('pde', 1.0))
    w_bc  = float(weights.get('bc',  1.0))
    w_ic  = float(weights.get('ic',  1.0))

    # canonicalize nu (accept legacy gamma)
    if dim==1 and not timep:
        a      = _as_fun(coeffs.get('a', 0.0), 1)
        nu_val = coeffs.get('nu', coeffs.get('gamma', 0.0))
        try:
            nu = float(nu_val)
        except Exception:
            nu = 0.0
        R  = _as_fun(forcing, 1)

    elif dim==1 and timep:
        a      = _as_fun(coeffs.get('a', 1.0), 1)
        nu     = float(coeffs.get('nu', coeffs.get('gamma', 0.0)))
        R      = _as_fun(forcing, 2)

    elif dim==2 and not timep:
        a      = _as_fun(coeffs.get('a', 0.0), 2)
        b      = _as_fun(coeffs.get('b', 0.0), 2)
        nu     = float(coeffs.get('nu', coeffs.get('gamma', 0.0)))
        R      = _as_fun(forcing, 2)

    elif dim==2 and timep:
        a      = _as_fun(coeffs.get('a', 0.0), 2)
        b      = _as_fun(coeffs.get('b', 0.0), 2)
        nu     = float(coeffs.get('nu', coeffs.get('gamma', 0.0)))
        R      = _as_fun(forcing, 3)
    else:
        raise ValueError("Unsupported (dim, time) combination")

    # --- random features (ELM) ---
    rng   = np.random.default_rng(seed)
    D     = dim + (1 if timep else 0)
    W_in  = rng.standard_normal((N_star, D))
    b_in  = rng.standard_normal((N_star, 1))

    # --- collocation sampling ---
    if dim == 1:
        xL, xR = domain.get('x', (0.0, 1.0))
        hal = Halton(1, scramble=False, seed=seed)
        x_f = xL + (xR - xL) * hal.random(N_f).reshape(-1, 1)

        if timep:
            tmax = float(problem.get('tmax', 0.5))
            t_f  = rng.uniform(0.0, tmax, size=(N_f, 1))

        bc_type = bc.get('type', 'dirichlet').lower()
        if bc_type == 'periodic':
            x_bc = None
        else:
            x_bc = np.array([[xL], [xR]])

        if timep:
            t_bc = np.linspace(0, tmax, N_bc).reshape(-1, 1)
            x_ic = np.linspace(xL, xR, max(2, N_ic)).reshape(-1, 1)
            t_ic = np.zeros_like(x_ic)

    else:   # --------------------- 2-D ----------------------------------------
        dtyp = domain.get('type', 'rect').lower()

        if dtyp == 'rect':
            xL, xR = domain.get('x', (0.0, 1.0))
            yL, yR = domain.get('y', (0.0, 1.0))
            hal    = Halton(2, scramble=False, seed=seed)
            pts    = hal.random(N_f)
            x_f    = xL + (xR - xL) * pts[:, 0:1]
            y_f    = yL + (yR - yL) * pts[:, 1:2]

            # boundary sampling on rectangle
            bc_pts = []
            perim  = 2 * (xR - xL) + 2 * (yR - yL)
            for i in range(N_bc):
                s = (i / max(1,N_bc)) * perim
                if   s <= (xR - xL):                                  # bottom
                    bc_pts.append([xL + s, yL])
                elif s <= (xR - xL) + (yR - yL):                       # right
                    bc_pts.append([xR, yL + (s - (xR - xL))])
                elif s <= 2 * (xR - xL) + (yR - yL):                   # top
                    bc_pts.append([xR - (s - (xR - xL) - (yR - yL)), yR])
                else:                                                  # left
                    bc_pts.append([xL, yR - (s - 2 * (xR - xL) - (yR - yL))])
            bc_pts = np.array(bc_pts, dtype=float)
            N_bc   = bc_pts.shape[0]

            if timep:
                tmax = float(problem.get('tmax', 0.5))
                t_f  = rng.uniform(0.0, tmax, size=(N_f, 1))
                t_bc = rng.uniform(0.0, tmax, size=(N_bc, 1))

        elif dtyp == 'polygon':
            verts = np.asarray(domain['verts'], dtype=float)
            path  = Path(verts)

            ngrid = int(np.sqrt(N_f) * 1.4)
            xs = ys = np.linspace(0, 1, max(20, ngrid))
            xx, yy = np.meshgrid(xs, ys)
            P_all  = np.c_[xx.ravel(), yy.ravel()]
            inside = path.contains_points(P_all)
            P_in   = P_all[inside]

            if len(P_in) < N_f:
                N_f = len(P_in)
            sel  = rng.choice(len(P_in), size=N_f, replace=False)
            x_f  = P_in[sel, 0:1]
            y_f  = P_in[sel, 1:2]

            # polygon boundary
            bc_pts = verts.copy()
            if len(bc_pts) < N_bc:
                need = N_bc - len(bc_pts)
                edge_pts = []
                for i in range(len(verts)):
                    v1, v2 = verts[i], verts[(i + 1) % len(verts)]
                    n_here = max(1, need // len(verts))
                    for j in range(n_here):
                        t = (j + 1) / (n_here + 1)
                        edge_pts.append((1 - t) * v1 + t * v2)
                        if len(edge_pts) >= need: break
                    if len(edge_pts) >= need: break
                if edge_pts:
                    bc_pts = np.vstack([bc_pts, np.array(edge_pts[:need])])

            bc_pts = bc_pts[:N_bc]
            N_bc   = bc_pts.shape[0]

            if timep:
                tmax = float(problem.get('tmax', 0.5))
                t_f  = rng.uniform(0.0, tmax, size=(N_f, 1))
                t_bc = rng.uniform(0.0, tmax, size=(N_bc, 1))
        else:
            raise ValueError("2D domain.type must be 'rect' or 'polygon'")
            
        if timep:
            if dtyp == 'rect':
                nx = int(np.sqrt(max(8, N_ic))); ny = nx
                xx0, yy0 = np.meshgrid(np.linspace(xL,xR,nx), np.linspace(yL,yR,ny))
                x_ic = xx0.ravel()[:,None]; y_ic = yy0.ravel()[:,None]
            else:
                sel = rng.choice(len(P_in), size=max(8, N_ic), replace=False)
                x_ic = P_in[sel,0:1]; y_ic = P_in[sel,1:2]
            t_ic = np.zeros_like(x_ic)

    # ---------- Build PDE rows ----------
    def _Z(P):     # P: (N, D)
        return P @ W_in.T + b_in.T

    rows = []
    rhs  = []

    if dim==1 and not timep:
        Zf = _Z(np.c_[x_f])
        m  = W_in[:,0:1]
        if optype in ('advection','adv','adv1d'):
            Hf = phi_p(Zf) * m.T * a(x_f)
            Kf = _col(R(x_f), Zf.shape[0])
        elif optype in ('poisson','diffusion','diff'):
            Hf = phi_pp(Zf) * (m.T**2)
            Kf = _col(R(x_f), Zf.shape[0])
        elif optype in ('advdiff','advection-diffusion','adv-diff'):
            Hf = phi_p(Zf) * m.T * a(x_f) + float(nu) * phi_pp(Zf) * (m.T**2)
            Kf = _col(R(x_f), Zf.shape[0])
        else:
            raise ValueError(f"Unsupported 1D steady operator '{optype}'")
        rows.append(w_pde*Hf); rhs.append(w_pde*Kf)

    elif dim==1 and timep:
        Zf = _Z(np.c_[x_f, t_f])
        m = W_in[:,0:1]     # x
        n = W_in[:,1:2]     # t
        Hf = phi_p(Zf) * n.T + phi_p(Zf) * m.T * a(x_f) + float(nu) * phi_pp(Zf) * (m.T**2)
        Kf = _col(R(x_f, t_f), Zf.shape[0])
        rows.append(w_pde*Hf); rhs.append(w_pde*Kf)

    elif dim==2 and not timep:
        Zf = _Z(np.c_[x_f, y_f])
        px = W_in[:,0:1]
        qy = W_in[:,1:2]
        adv  = phi_p(Zf) * px.T * a(x_f, y_f) + phi_p(Zf) * qy.T * b(x_f, y_f)
        diff = float(nu) * phi_pp(Zf) * (px.T**2 + qy.T**2)
        Hf   = adv + diff
        Kf   = _col(R(x_f, y_f), Zf.shape[0])
        rows.append(w_pde*Hf); rhs.append(w_pde*Kf)

    elif dim==2 and timep:
        Zf = _Z(np.c_[x_f, y_f, t_f])
        px = W_in[:,0:1]
        qy = W_in[:,1:2]
        rt = W_in[:,2:3]
        adv  = phi_p(Zf) * rt.T + phi_p(Zf) * px.T * a(x_f, y_f) + phi_p(Zf) * qy.T * b(x_f, y_f)
        diff = float(nu) * phi_pp(Zf) * (px.T**2 + qy.T**2)
        Hf   = adv + diff
        Kf   = _col(R(x_f, y_f, t_f), Zf.shape[0])
        rows.append(w_pde*Hf); rhs.append(w_pde*Kf)

    # ---------- Boundary rows ----------
    bc_type = bc.get('type','dirichlet').lower()
    if dim==1:
        if bc_type == 'periodic':
            if timep:
                xL_arr = np.full((N_bc,1), xL); xR_arr = np.full((N_bc,1), xR)
                ZL = _Z(np.c_[xL_arr, t_bc]); ZR = _Z(np.c_[xR_arr, t_bc])
                Hbc = (phi(ZL) - phi(ZR)); Kbc = np.zeros((N_bc,1))
            else:
                xL_arr = np.full((N_bc,1), xL); xR_arr = np.full((N_bc,1), xR)
                ZL = _Z(np.c_[xL_arr]); ZR = _Z(np.c_[xR_arr])
                Hbc = (phi(ZL) - phi(ZR)); Kbc = np.zeros((N_bc,1))
            rows.append(w_bc*Hbc); rhs.append(w_bc*_col(Kbc, N_bc))

        elif bc_type == 'dirichlet':
            gL = bc.get('gL', 0.0); gR = bc.get('gR', 0.0)
            gL = _as_fun(gL, 2 if timep else 1)
            gR = _as_fun(gR, 2 if timep else 1)
            if timep:
                ZL = _Z(np.c_[np.full_like(t_bc, xL), t_bc])
                ZR = _Z(np.c_[np.full_like(t_bc, xR), t_bc])
                rows.append(w_bc*phi(ZL)); rhs.append(w_bc*_col(gL(np.full_like(t_bc, xL), t_bc), len(t_bc)))
                rows.append(w_bc*phi(ZR)); rhs.append(w_bc*_col(gR(np.full_like(t_bc, xR), t_bc), len(t_bc)))
            else:
                nL = max(1, (N_bc + 1) // 2)
                nR = max(1, N_bc // 2)
                xL_arr = np.full((nL,1), xL)
                xR_arr = np.full((nR,1), xR)
                ZL = _Z(np.c_[xL_arr]); ZR = _Z(np.c_[xR_arr])
                rows.append(w_bc*phi(ZL)); rhs.append(w_bc*_col(gL(xL_arr), nL))
                rows.append(w_bc*phi(ZR)); rhs.append(w_bc*_col(gR(xR_arr), nR))

        elif bc_type == 'neumann':
            gL = _as_fun(bc.get('gL', 0.0), 1)
            gR = _as_fun(bc.get('gR', 0.0), 1)
            m  = W_in[:,0:1].T
            if timep:
                ZL = _Z(np.c_[np.full_like(t_bc,xL), t_bc])
                ZR = _Z(np.c_[np.full_like(t_bc,xR), t_bc])
                rows.append(w_bc*phi_p(ZL)*m); rhs.append(w_bc*_col(gL(np.full_like(t_bc,xL)), len(t_bc)))
                rows.append(w_bc*phi_p(ZR)*m); rhs.append(w_bc*_col(gR(np.full_like(t_bc,xR)), len(t_bc)))
            else:
                nL = max(1, (N_bc + 1) // 2)
                nR = max(1, N_bc // 2)
                xL_arr = np.full((nL,1), xL)
                xR_arr = np.full((nR,1), xR)
                ZL = _Z(np.c_[xL_arr]); ZR = _Z(np.c_[xR_arr])
                rows.append(w_bc*phi_p(ZL)*m); rhs.append(w_bc*_col(gL(xL_arr), nL))
                rows.append(w_bc*phi_p(ZR)*m); rhs.append(w_bc*_col(gR(xR_arr), nR))
        else:
            raise ValueError("1D bc.type must be 'dirichlet' | 'neumann' | 'periodic'")

    else:
        if bc_type != 'dirichlet':
            raise ValueError("2D: only Dirichlet BC supported")
        g_val = bc.get('g', 0.0)
        Zb = _Z(np.c_[bc_pts, t_bc]) if timep else _Z(np.c_[bc_pts])
        if callable(g_val):
            if timep:
                Kb = g_val(bc_pts[:,0:1], bc_pts[:,1:2], t_bc)
            else:
                Kb = g_val(bc_pts[:,0:1], bc_pts[:,1:2])
        else:
            Kb = np.full((N_bc, 1), float(g_val))
        rows.append(w_bc*phi(Zb)); rhs.append(w_bc*_col(Kb, N_bc))

    # ---------- Initial condition rows ----------
    if timep:
        if dim==1:
            Zi = _Z(np.c_[x_ic, t_ic])
            Fi = _as_fun(ic['F'], 1)
            ic_val = Fi(x_ic)
            rows.append(w_ic*phi(Zi)); rhs.append(w_ic*_col(ic_val, len(x_ic)))
        else:
            Zi = _Z(np.c_[x_ic, y_ic, t_ic])
            Fi = _as_fun(ic['F'], 2)
            ic_val = Fi(x_ic, y_ic)
            rows.append(w_ic*phi(Zi)); rhs.append(w_ic*_col(ic_val, len(x_ic)))

    # ---------- stack & solve ----------
    H = np.vstack(rows)            # (N_rows, N_star)
    K = np.vstack(rhs)             # list of (Ni,1) -> (N_rows,1)
    K = _col(K, H.shape[0])        # ensure (N_rows, 1)

    if ridge and ridge > 0.0:
        M = H.shape[1]
        c = np.linalg.solve(H.T @ H + ridge*np.eye(M), H.T @ K)
    else:
        c = np.linalg.pinv(H, rcond=rcond) @ K

    # ---------- predictor ----------
    def _u_from_P(P):
        return (phi(P @ W_in.T + b_in.T) @ c).ravel()

    if dim==1 and not timep:
        def u_pred(x):
            x = np.asarray(x).reshape(-1,1)
            P = np.c_[x]
            return _u_from_P(P)
    elif dim==1 and timep:
        def u_pred(x, t):
            x = np.asarray(x).reshape(-1,1)
            t = np.asarray(t).reshape(-1,1)
            P = np.c_[x, t]
            return _u_from_P(P)
    elif dim==2 and not timep:
        def u_pred(P_or_xy, y=None):
            if y is None:
                P = np.asarray(P_or_xy)
                if P.ndim==1: P = P.reshape(1,-1)
                return _u_from_P(P)
            else:
                x = np.asarray(P_or_xy).reshape(-1,1)
                y = np.asarray(y).reshape(-1,1)
                P = np.c_[x, y]
                return _u_from_P(P)
    else:
        def u_pred(P_or_x, y=None, t=None):
            if y is None and t is None:
                P = np.asarray(P_or_x)
                if P.ndim==1: P = P.reshape(1,-1)
                return _u_from_P(P)
            else:
                x = np.asarray(P_or_x).reshape(-1,1)
                y = np.asarray(y).reshape(-1,1)
                t = np.asarray(t).reshape(-1,1)
                P = np.c_[x, y, t]
                return _u_from_P(P)

    # ---------- plotting ----------
    if do_plot:
        if dim==1 and not timep:
            xL, xR = domain.get('x',(0.0,1.0))
            _plot1d_steady(u_pred, exact, xL, xR, n_plot, title=optype.upper())
        elif dim==1 and timep:
            xL, xR = domain.get('x',(0.0,1.0))
            _plot1d_unsteady(u_pred, exact, xL, xR, times, n_plot, title=optype.upper())
        elif dim==2 and not timep:
            dtyp = domain.get('type','rect').lower()
            if dtyp == 'rect':
                xL, xR = domain.get('x',(0.0,1.0)); yL, yR = domain.get('y',(0.0,1.0))
                _plot2d_rect_steady(u_pred, exact, xL, xR, yL, yR, n_plot, title=optype.upper())
                if surf3d:
                    plot_3d_surface(u_pred, exact, xL, xR, yL, yR, 50, title=optype.upper())
            else:
                _plot2d_poly_steady(u_pred, exact, Path(domain['verts']), n_plot, title=optype.upper())
        else:
            xL, xR = domain.get('x',(0.0,1.0)); yL, yR = domain.get('y',(0.0,1.0))
            _plot2d_unsteady(u_pred, exact, xL, xR, yL, yR, times, n_plot, title=optype.upper())
            _plot2d_unsteady_error(u_pred, exact, xL, xR, yL, yR, times, n_plot, title=optype.upper())
            if surf3d:
                plot_3d_unsteady(u_pred, exact, xL, xR, yL, yR, times, 50, title=optype.upper())

    info = {
        'N_f': N_f, 'N_bc': N_bc, 'N_ic': N_ic, 'N_star': N_star,
        'seed': seed, 'ridge': ridge, 'rcond': rcond,
        'W_in': W_in, 'b_in': b_in, 'c': c, 'dim': dim, 'time': timep,
        'optype': optype
    }
    return u_pred, info
