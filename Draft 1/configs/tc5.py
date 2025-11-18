import numpy as np

nu = -0.05  # negative encodes u_t - 0.05 u_xx = 0
u_exact = lambda x, t: np.sin(np.pi*x) * np.exp(-(abs(nu))*(np.pi**2)*t)
forcing  = lambda x, t: np.zeros_like(x)
xL, xR = 0.0, 1.0

problem = {
    'dim': 1, 'time': True,
    'operator': {'type': 'diff'},
    'coeffs': {'a': 0.0, 'nu': nu},
    'forcing': forcing,
    'domain': {'type':'interval','x':(xL,xR)},
    'bc': {'type':'dirichlet', 'gL': 0.0, 'gR': 0.0},
    'ic': {'F': lambda x: u_exact(x, 0.0)},
    'tmax': 1.0,
    'collocation': {'N_f': 1200, 'N_bc': 120, 'N_ic': 120},
    'model': {'N_star': 1600, 'seed': 8, 'ridge': 1e-8},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 900, 'times': [0.0, 0.25, 0.5, 0.75, 1.0]}
}
