
from solver import LBMFlowSolver
import jax

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    solver = LBMFlowSolver()
    solver.run_simulation()
