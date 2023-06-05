import jax.numpy as jnp


def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)  # last axis
    return density


class LBMSolver:
    def __init__(self, nx, ny, tau, vel = 0.04, n_discrete_velocities = 9):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.vel = vel
        self.n_discrete_velocities = n_discrete_velocities

        self.init_fields()

    def init_fields(self):
        self.fin = jnp.zeros((self.nx, self.ny, self.n_discrete_velocities))
        self.fout = jnp.zeros((self.nx, self.ny, self.n_discrete_velocities))
        self.rho = jnp.zeros((self.nx, self.ny))
        self.u = jnp.zeros((self.nx, self.ny, 2))

    def collide_and_stream(self):
        # This method should implement the collide and stream step of the LBM
        pass

    def apply_boundary_conditions(self):
        # This method should apply appropriate boundary conditions
        pass

    def compute_macroscopic_variables(self):
        # This method should compute the macroscopic variables (density and velocity)


        pass

    def solve(self, nsteps):
        for step in range(nsteps):
            self.collide_and_stream()
            self.apply_boundary_conditions()
            self.compute_macroscopic_variables()
