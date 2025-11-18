import numpy as np
pi = np.pi

a, b, nu = 0.5, 0.3, -0.01

def u_xy(x, y): return np.sin(np.pi*x) * np.sin(np.pi*y)
def u_exact(x, y, t): return u_xy(x, y) * np.exp(-t)

def forcing(x, y, t):
    ut  = -u_xy(x,y) * np.exp(-t)
    ux  =  np.pi*np.cos(np.pi*x) * np.sin(np.pi*y) * np.exp(-t)
    uy  =  np.pi*np.sin(np.pi*x) * np.cos(np.pi*y) * np.exp(-t)
    uxx = - (np.pi**2) * u_xy(x,y) * np.exp(-t)
    uyy = - (np.pi**2) * u_xy(x,y) * np.exp(-t)
    return ut + a*ux + b*uy + nu*(uxx + uyy)  # +nu form; nu negative

plot_times = [0.0, 0.3, 0.6, 1.0]

problem = {
    'dim': 2, 'time': True,
    'operator': {'type': 'advdiff'},
    'coeffs': {'a': a, 'b': b, 'nu': nu},
    'forcing': forcing,
    'domain': {'type':'rect','x':(0.0,1.0),'y':(0.0,1.0)},
    'bc': {'type': 'dirichlet', 'g': lambda x,y,t: u_exact(x,y,t)},
    'ic': {'F': lambda x,y: u_exact(x,y,0.0)},
    'tmax': 1.0,
    'collocation': {'N_f': 6000, 'N_bc': 600, 'N_ic': 600},
    'model': {'N_star': 7000, 'seed': 11, 'ridge': 1e-8},
    'exact': u_exact,
    'plot': {'enabled': True, 'n_plot': 120, 'times': plot_times, 'surface3d': True}
}
