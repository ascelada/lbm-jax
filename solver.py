import flax as flax
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from functions import update
from domain import generate_lattice_spheres

import porespy as ps
@flax.struct.dataclass
class LBMFlowSolver:
    jax.config.update("jax_enable_x64", True)

    N_DISCRETE_VELOCITIES = 9
    N_ITERATIONS = 15_000
    REYNOLDS_NUMBER = 80
    DIAMETER = 1
    NX = 500
    NY = 500
    MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04
    SAVE_N_STEPS_TRUE = 1000
    PLOT_N_STEPS_TRUE = 500
    ANIMATE = False
    SKIP_FIRST_N_ITERATIONS = 0
    VISUALIZE = True
    KINEMATIC_VISCOSITY = 0.01
    RHO = 1
    RELAXATION_OMEGA = 1.0 / (3.0 * KINEMATIC_VISCOSITY + 0.5)
    velocity_profile = jnp.zeros((NX, NY, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HORIZONTAL_INFLOW_VELOCITY)
    x = jnp.arange(NX)
    y = jnp.arange(NY)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    LATTICE_VELOCITIES_X = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1, ])
    LATTICE_VELOCITIES_Y = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1, ])
    LATTICE_VELOCITIES = jnp.array([
        LATTICE_VELOCITIES_X,
        LATTICE_VELOCITIES_Y, ]
    )
    LATTICE_INDICES = jnp.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    OPPOSITE_LATTICE_INDICES = jnp.array(
        [0, 3, 4, 1, 2, 7, 8, 5, 6]
    )
    LATTICE_WEIGHTS = jnp.array([
        4 / 9,
        1 / 9, 1 / 9, 1 / 9, 1 / 9,
        1 / 36, 1 / 36, 1 / 36, 1 / 36
    ])
    RIGHT_VELOCITIES = jnp.array([1, 5, 8])
    UP_VELOCITIES = jnp.array([2, 5, 6])
    LEFT_VELOCITIES = jnp.array([3, 6, 7])
    DOWN_VELOCITIES = jnp.array([4, 7, 8])
    PURE_VERTICAL_VELOCITIES = jnp.array([0, 2, 4])
    PURE_HORIZONTAL_VELOCITIES = jnp.array([0, 1, 3])
    isPorous = True
    mask = jnp.array(~ps.generators.lattice_spheres(shape=[NX, NY],lattice="tri",r= 12, spacing= 50, offset= 15))
    acceleration_x = 0.0001
    ACCELERATION_MASK = jnp.where(mask,0.0, acceleration_x)
    porosity = jnp.sum(mask)/(NX*NY)
    @classmethod
    def config(cls, rho,inflow_vel, kinematic_viscosity, nx, ny):

        cls.RHO = rho
        cls.KINEMATIC_VISCOSITY = kinematic_viscosity
        cls.REYNOLDS_NUMBER = inflow_vel*ny/kinematic_viscosity
        cls.NX = nx
        cls.NY = ny
        cls.MAX_HORIZONTAL_INFLOW_VELOCITY = inflow_vel
        cls.RELAXATION_OMEGA = 1.0 / (3.0 * kinematic_viscosity + 0.5)


        print("Omega: ", cls.RELAXATION_OMEGA)
        print("Omega should be 0.5<=omega<1.5")
        print("Reynolds: ", cls.REYNOLDS_NUMBER)

        response = input("Do You Want To Continue? [y/n]").lower().strip()
        if response != 'y':
            print("Exiting program")
            exit()
        return cls

    @staticmethod
    def get_density(discrete_velocities):
        """
        Calculates the density of the fluid by summing up the discrete velocities along the last axis.

        :param discrete_velocities: Discrete velocities for each direction. Shape: (NX, NY, N_DISCRETE_VELOCITIES)
        :return: Density of the fluid. Shape: (NX, NY)
        """

        density = jnp.sum(discrete_velocities, axis=-1)  # sum along the last axis
        return density

    def get_macroscopic_velocities(self, discrete_velocities, density):
        """
        Calculates the macroscopic velocities for each grid cell (x, y) by taking a weighted sum of the discrete velocities
        and dividing by the total density.

        :param discrete_velocities: Discrete velocities for each direction. Shape: (NX, NY, N_DISCRETE_VELOCITIES)
        :param density: Density of the fluid. Shape: (NX, NY)
        :return: Macroscopic velocities of the fluid. Shape: (NX, NY, 2)
        """
        macroscopic_velocities = jnp.einsum(
            "NMQ,dQ->NMd",
            discrete_velocities,
            self.LATTICE_VELOCITIES
        ) / density[..., jnp.newaxis]
        return macroscopic_velocities

    def get_equilibrium_velocities(self,macroscopic_velocities, density):
        """
        Calculates the equilibrium discrete velocities for each grid cell (x, y) based on the macroscopic velocities
        and total density. The equilibrium discrete velocities are derived from the equilibrium distribution function,
        which is an approximation of the Maxwell-Boltzmann distribution.
        Args:
            macroscopic_velocities: Macroscopic velocities for each grid cell. Shape: (NX, NY, 2)
            density: Density of the fluid. Shape: (NX, NY)

        Returns:
            Equilibrium discrete velocities for each grid cell in each direction.
            Shape: (NX, NY, N_DISCRETE_VELOCITIES)
        """

        # Project macroscopic velocities onto the lattice velocities
        projected_discrete_velocities = jnp.einsum(
            "dQ,NMd->NMQ",
            self.LATTICE_VELOCITIES,
            macroscopic_velocities
        )

        # Calculate the magnitude of the macroscopic velocities
        macroscopic_velocity_magnitude = jnp.linalg.norm(
            macroscopic_velocities, axis=-1, ord=2
        )

        # Calculate the equilibrium discrete velocities based on the equilibrium distribution function
        equilibrium_discrete_velocities = (
                density[..., jnp.newaxis]
                *
                self.LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
                *
                (
                        1
                        +
                        3 * projected_discrete_velocities
                        +
                        9 / 2 * projected_discrete_velocities ** 2
                        -
                        3 / 2 * macroscopic_velocity_magnitude[..., jnp.newaxis] ** 2
                )
        )
        return equilibrium_discrete_velocities

    def compute_velocity(self, data):
        """
        Compute the magnitude of the macroscopic velocities for each point in the grid.

        The magnitude is calculated as the Euclidean norm of the velocity vector at each point.

        Args:
            data: An array representing the grid. It has the shape (NX, NY, N_DISCRETE_VELOCITIES),
                  where NX is the number of points in the x direction,
                  NY is the number of points in the y direction,
                  and N_DISCRETE_VELOCITIES is the number of discrete velocities in the LBM.

        Returns:
            An array of shape (NX, NY) where each element represents the magnitude of the macroscopic
            velocity at a point in the grid.
        """
        density = self.get_density(data)
        macroscopic_velocities = self.get_macroscopic_velocities(data, density)
        velocity_magnitude_sq = jnp.einsum('ijk,ijk->ij', macroscopic_velocities, macroscopic_velocities)
        velocity_magnitude = jnp.sqrt(velocity_magnitude_sq)
        return velocity_magnitude


    def run(self, discrete_velocities_prev):
        with h5py.File('data.hdf5', 'w') as raw:
            raw_group = raw.create_group('raw_data')
            vel_group = raw.create_group('vel_data')
            mask_group = raw.create_group('mask_data')
            mask_group.create_dataset('mask_data',data=self.mask)


            for iteration_idx in tqdm(range(self.N_ITERATIONS)):
                # If it's the first iteration or the current iteration is a multiple of SAVE_N_STEPS_TRUE,
                # save the current state of the simulation.
                if iteration_idx == 0 or iteration_idx % self.SAVE_N_STEPS_TRUE == 0:
                    data_numpy = np.array(discrete_velocities_prev)
                    vel_data = np.array(self.compute_velocity(discrete_velocities_prev))

                    raw_group.create_dataset(f'timestep_{iteration_idx}',
                                             data=data_numpy,
                                             compression='gzip',
                                             compression_opts=5)

                    vel_group.create_dataset(f'timestep_{iteration_idx}',
                                             data=vel_data,
                                             compression='gzip',
                                             compression_opts=5)

                # Perform an update step in the simulation
                discrete_velocities_next = update(self,discrete_velocities_prev)
                discrete_velocities_prev = discrete_velocities_next

                # If the current iteration is a multiple of PLOT_N_STEPS_TRUE, and the iteration is higher than SKIP_FIRST_N_ITERATIONS,
                # plot the current state of the simulation.
                if iteration_idx % self.PLOT_N_STEPS_TRUE == 0 and self.VISUALIZE and iteration_idx > self.SKIP_FIRST_N_ITERATIONS:
                    self.plot(discrete_velocities_next)

    def plot(self, data):
        """
        This function generates plots of the velocity field, vorticity, and
        velocity profile along the vertical centerline of the lattice.

        Args:
            data (jnp.array): The distribution function, which provides the
                              probability of finding a particle with a certain
                              velocity at a given point in the lattice.
        """
        density = self.get_density(data)
        macroscopic_velocities = (self.get_macroscopic_velocities(
            data,
            density
        ))
        velocity_magnitude = jnp.linalg.norm(
            macroscopic_velocities,
            axis=-1,
            ord=2,
        )

        # d_u__d_x, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
        # d_v__d_x, d_v__d_y = jnp.gradient(macroscopic_velocities[..., 1])
        #
        # curl = (d_u__d_y - d_v__d_x)

        X_masked = self.X[self.mask]
        Y_masked = self.Y[self.mask]

        plt.subplot()
        plt.contourf(
            self.X,
            self.Y,
            velocity_magnitude,
            cmap=cmr.lavender,
            levels=50,
        )
        plt.scatter(
            X_masked,
            Y_masked,
            c='#C3c932',
            s=10,
        )
        plt.colorbar().set_label("Velocity Magnitude")

        # plt.subplot(212)
        # plt.contourf(
        #     self.X,
        #     self.Y,
        #     curl,
        #     levels=50,
        #     vmin=-0.02,
        #     vmax=0.02,
        #     cmap=cmr.eclipse
        # )
        # plt.scatter(
        #     X_masked,
        #     Y_masked,
        #     c='#339C4D',  # adjust color as needed
        #     s=10,  # adjust size as needed
        # )
        # plt.colorbar().set_label("Vorticity Magnitude")
        plt.rcParams["figure.figsize"] = (10, 5)

        #plt.subplot(313)
        #plt.plot(self.y[1:(self.NY - 1)], velocity_magnitude[self.NX // 2, 1:-1])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        plt.show()
    def find_entrance_length(self,velocity_matrix, threshold):
        # Iterate over the rows in the velocity matrix
        for i, velocity_profile in enumerate(velocity_matrix):
            # If the standard deviation of the velocities is below the threshold, return the index
            if jnp.std(velocity_profile) < threshold:
                return i
        # If no entrance length was found, return a sentinel value
        return -1
    def animate(self):
        file = h5py.File('data.hdf5', 'r')
        g1 = file.get('vel_data')

        fig, ax = plt.subplots()

        keys = sorted(g1.keys(), key=lambda x: int(x.split('_')[1]))

        initial_data = g1[keys[0]][:]

        contour = ax.contourf(self.X, self.Y, initial_data, cmap=cmr.lavender,
                              levels=50, )

        fig.colorbar(contour)

        def updatefig(key):
            data = g1[key][:]
            contour = ax.contourf(self.X, self.Y, data, cmap=cmr.lavender,
                                  levels=50, )
            return contour,

        ani = FuncAnimation(fig, updatefig, frames=keys, interval=50, blit=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)

        ani.save('animation.mp4', writer=writer)
        file.close()
    def run_simulation(self):

        self.run(self.get_equilibrium_velocities(jnp.zeros((self.NX, self.NY, 2)), jnp.ones((self.NX, self.NY))))
        if self.ANIMATE:
            self.animate()