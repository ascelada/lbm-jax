import analysis
from domain import visualize_labeled_matrix
from solver import LBMFlowSolver
import jax

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    solver = LBMFlowSolver()

    print("Omega: ", solver.RELAXATION_OMEGA)
    print("Omega should be 0.5<=omega<1.5")
    print("Reynolds: ", solver.REYNOLDS_NUMBER)
    print("Porosity: ", solver.porosity)
    visualize_labeled_matrix(solver.mask)

    response = input("Do You Want To Continue? [y/n]").lower().strip()
    if response != 'y':
        print("Exiting program")
        exit()
    solver.run_simulation()
    #print(analysis.compare_entrance_length('data.hdf5', 'vel_data',0.001,50,1000,0.001))
