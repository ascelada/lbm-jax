import analysis
from domain import visualize_labeled_matrix, label_islands
from solver import LBMFlowSolver
import numpy as np
import jax

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    domain = np.load('matrix_5.npy')
    solver = LBMFlowSolver.config(domain)

    print("Omega: ", solver.RELAXATION_OMEGA)
    print("Omega should be 0.5<=omega<1.5")
    print("Reynolds: ", solver.REYNOLDS_NUMBER)
    print("Porosity: ", solver.porosity)
    # labels, positions = label_islands(solver.mask)
    visualize_labeled_matrix(solver.mask)

    # response = input("Do You Want To Continue? [y/n]").lower().strip()
    # if response != 'y':
    #     print("Exiting program")
    #     exit()
    solver.run_simulation()

