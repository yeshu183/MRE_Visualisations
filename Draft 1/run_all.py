# run_all.py
from importlib import import_module
from PIELM_solver import solve_pielm

modules = [f"configs.tc{i}" for i in range(9, 11)]

for modname in modules:
    print(f"\n================ {modname} ================")
    cfg = import_module(modname)
    solve_pielm(cfg.problem)
