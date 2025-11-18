import numpy as np

nu = -0.20  # NEGATIVE -> a u_x - |nu| u_xx = 0
u_exact = lambda x: (np.exp(x/(-nu)) - 1.0) / (np.exp(1.0/(-nu)) - 1.0)
R = lambda x: np.zeros_like(x)
xL, xR = 0.0, 1.0
gL, gR = 0.0, 1.0

problem = {
    'dim': 1, 'time': False,
    'operator': {'type': 'advdiff'},
    'coeffs': {'a': 1.0, 'nu': float(nu)},
    'forcing': R,
    'domain': {'type':'interval','x':(xL,xR)},
    'bc': {'type':'dirichlet','gL': gL, 'gR': gR},
    'collocation': {'N_f': 120, 'N_bc': 2},
    'model': {'N_star': 122, 'seed': 3, 'ridge': 0.0},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 900}
}
