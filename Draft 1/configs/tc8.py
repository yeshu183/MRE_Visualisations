import numpy as np
pi = np.pi

a, b, nu = 1.0, 1.0, 0.01

def u_exact(x, y):
    return np.sin(np.pi*x) * np.sin(np.pi*y)

def forcing(x, y):
    ux  = np.pi*np.cos(np.pi*x) * np.sin(np.pi*y)
    uy  = np.pi*np.sin(np.pi*x) * np.cos(np.pi*y)
    uxx = - (np.pi**2) * np.sin(np.pi*x) * np.sin(np.pi*y)
    uyy = - (np.pi**2) * np.sin(np.pi*x) * np.sin(np.pi*y)
    return a*ux + b*uy + nu*(uxx + uyy)   # +nu form

problem = {
    'dim': 2, 'time': False,
    'operator': {'type': 'advdiff'},
    'coeffs': {'a': a, 'b': b, 'nu': nu},
    'forcing': forcing,
    'domain': {'type':'rect','x':(0.0,1.0),'y':(0.0,1.0)},
    'bc': {'type': 'dirichlet', 'g': u_exact},
    'collocation': {'N_f': 2500, 'N_bc': 400},
    'model': {'N_star': 3400, 'seed': 5, 'ridge': 1e-8},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 140, 'surface3d': True}
}
