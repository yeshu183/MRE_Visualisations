import numpy as np
pi = np.pi

a, nu = 0.7, -0.02
def u_exact(x, t):
    return np.sin(np.pi*x) * np.exp(-t)

def forcing(x, t):
    ut  = -np.sin(np.pi*x) * np.exp(-t)
    ux  =  np.pi*np.cos(np.pi*x) * np.exp(-t)
    uxx = - (np.pi**2) * np.sin(np.pi*x) * np.exp(-t)
    return ut + a*ux + nu*uxx    # +nu form (nu negative)

xL, xR = 0.0, 1.0

problem = {
    'dim': 1, 'time': True,
    'operator': {'type': 'advdiff'},
    'coeffs': {'a': a, 'nu': nu},
    'forcing': forcing,
    'domain': {'type':'interval','x':(xL,xR)},
    'bc': {'type':'dirichlet', 'gL': 0.0, 'gR': 0.0},
    'ic': {'F': lambda x: u_exact(x, 0.0)},
    'tmax': 1.0,
    'collocation': {'N_f': 1400, 'N_bc': 120, 'N_ic': 160},
    'model': {'N_star': 1800, 'seed': 4, 'ridge': 1e-8},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 900, 'times': [0.0, 0.3, 0.6, 1.0]}
}
