import numpy as np
pi = np.pi

def u_exact(x, y):
    return np.sin(np.pi*x) * np.sin(np.pi*y)

def forcing(x, y):  # u_xx + u_yy = -2π^2 u
    return -2*(np.pi**2) * np.sin(np.pi*x) * np.sin(np.pi*y)

problem = {
    'dim': 2, 'time': False,
    'operator': {'type': 'diff'},
    'coeffs': {'a': 0.0, 'b': 0.0, 'nu': 1.0},  # +nu
    'forcing': forcing,
    'domain': {'type':'rect','x':(0.0,1.0),'y':(0.0,1.0)},
    'bc': {'type': 'dirichlet', 'g': lambda x,y: 0.0},
    'collocation': {'N_f': 2500, 'N_bc': 400},
    'model': {'N_star': 3200, 'seed': 3, 'ridge': 1e-8},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 140, 'surface3d': True}
}
