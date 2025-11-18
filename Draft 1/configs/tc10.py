import numpy as np

def gaussian_ic(x, y):
    return np.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))

problem = {
    'dim': 2,
    'time': True,
    'operator': {'type': 'advdiff'},
    'coeffs': {'a': 1.0, 'b': -0.5, 'nu': 0.01},
    'forcing': lambda x, y, t: np.zeros_like(x),  # zero forcing
    'domain': {'type': 'rect', 'x': (0.0, 1.0), 'y': (0.0, 1.0)},
    'bc': {'type': 'dirichlet', 'g': 0.0},
    'ic': {'F': gaussian_ic},
    'tmax': 2.0,
    'collocation': {'N_f': 8000, 'N_bc': 800, 'N_ic': 800},
    'model': {'N_star': 9600, 'seed': 123, 'ridge': 1e-8},
    'plot': {'enabled': True, 'n_plot': 120, 'times': [0.0, 0.5, 1.0, 1.5, 2.0], 'surface3d': True}
}
