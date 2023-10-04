import analysis
from solver import LBMFlowSolver
import jax

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    solver = LBMFlowSolver()
    solver.run_simulation()
    #print(analysis.compare_entrance_length('data.hdf5', 'vel_data',0.001,50,1000,0.001))
