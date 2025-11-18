import numpy as np
pi = np.pi

u_exact = lambda x: np.sin(0.5*pi*x) * np.cos(2*pi*x) + 1.0
R = lambda x: - (17.0/4.0)*(pi**2)*np.sin(0.5*pi*x)*np.cos(2*pi*x) \
               - 2.0*(pi**2)*np.cos(0.5*pi*x)*np.sin(2*pi*x)
xL, xR = 0.0, 1.0
gL = float(u_exact(np.array([[xL]])).item())
gR = float(u_exact(np.array([[xR]])).item())

problem = {
    'dim': 1, 'time': False,
    'operator': {'type': 'diff'},  # or 'poisson'
    'coeffs': {},                  # nu ignored in 1D steady diff mode
    'forcing': R,
    'domain': {'type':'interval','x':(xL,xR)},
    'bc': {'type':'dirichlet','gL': gL, 'gR': gR},
    'collocation': {'N_f': 140, 'N_bc': 2},
    'model': {'N_star': 142, 'seed': 2, 'ridge': 0.0},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 900}
}
