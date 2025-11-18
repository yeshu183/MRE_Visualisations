import numpy as np

a = 1.0
u_exact = lambda x, t: 1.0 + np.sin(2*np.pi*(x - t))
forcing  = lambda x, t: np.zeros_like(x)
xL, xR = 0.0, 1.0

problem = {
    'dim': 1, 'time': True,
    'operator': {'type': 'adv'},
    'coeffs': {'a': a, 'nu': 0.0},
    'forcing': forcing,
    'domain': {'type':'interval','x':(xL,xR)},
    'bc': {'type':'periodic'},
    'ic': {'F': lambda x: u_exact(x, 0.0)},
    'tmax': 1.0,
    'collocation': {'N_f': 1200, 'N_bc': 200, 'N_ic': 180},
    'model': {'N_star': 1600, 'seed': 7, 'ridge': 1e-8},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 900, 'times': [0.0, 0.25, 0.5, 0.75, 1.0]}
}
