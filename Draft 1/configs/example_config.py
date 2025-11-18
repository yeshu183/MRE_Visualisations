import numpy as np
pi = np.pi

u_exact = lambda x: np.sin(2*pi*x) * np.cos(4*pi*x) + 1.0
R       = lambda x: 2*pi*np.cos(2*pi*x)*np.cos(4*pi*x) - 4*pi*np.sin(2*pi*x)*np.sin(4*pi*x)
xL, xR  = 0.0, 1.0
gL = float(u_exact(np.array([[xL]])).item())
gR = float(u_exact(np.array([[xR]])).item())

problem = {
    'dim': 1, 'time': False,
    'operator': {'type': 'adv'},
    'coeffs': {'a': 1.0},
    'forcing': R,
    'domain': {'type':'interval','x':(xL,xR)},
    'bc': {'type':'dirichlet','gL': gL, 'gR': gR},
    'collocation': {'N_f': 120, 'N_bc': 2},
    'model': {'N_star': 122, 'seed': 1, 'ridge': 0.0},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 900}
}
