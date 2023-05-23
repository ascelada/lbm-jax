# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cmasher as cmr
from tqdm import tqdm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    N_ITERATIONS = 15_000
    REYNOLDS_NUMBER = 80
    DIAMETER = 1

    NX = 300
    NY = 50
    MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04

    PLOT_N_STEPS_TRUE = 100
    SKIP_FIRST_N_ITERATIONS = 0
    VISUALIZE = True

    """
    LBM GRID = D2Q9
    """

    """
    macroscopic velocity : (Nx,Ny, 2)
    discrete velocity: Nx,Ny,9
    density (nx,ny)

    Re = U * D / v

    tao = 1/(3v+0.5)


    density: soma(fi)
    velocites: 1/p soma(fi c1)

    equilibrio:
    fie: p wi (1+3c* u +9/2 (c*u)^2 =3/2 
    """

    N_DISCRETE_VELOCITIES = 9

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

    kinematic_viscosity = 0.01
    RELAXATION_OMEGA = 1.0 / (
            3.0 *
            kinematic_viscosity
            +
            0.5)

    print(RELAXATION_OMEGA)

    velocity_profile = jnp.zeros((NX, NY, 2))
    velocity_profile = velocity_profile.at[0, :, 0].set(MAX_HORIZONTAL_INFLOW_VELOCITY)

    def get_density(discrete_velocities):
        density = jnp.sum(discrete_velocities, axis=-1)  # last axis
        return density


    def get_macroscopic_velocities(discrete_velocities, density):
        macroscopic_velocities = jnp.einsum(
            "NMQ,dQ->NMd",
            discrete_velocities,
            LATTICE_VELOCITIES
        ) / density[..., jnp.newaxis]
        return macroscopic_velocities


    def get_equilibrium_velocities(macroscopic_velocities, density):
        projected_discrete_velocities = jnp.einsum(
            "dQ,NMd->NMQ",
            LATTICE_VELOCITIES,
            macroscopic_velocities
        )
        macroscopic_velocity_magnitude = jnp.linalg.norm(
            macroscopic_velocities, axis=-1, ord=2
        )
        equilibrium_discrete_velocities = (
                density[..., jnp.newaxis]
                *
                LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
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


    @jax.jit
    def update(discrete_velocities_prev):
        # 1 Prescribe outflow on right boundary
        discrete_velocities_prev = discrete_velocities_prev.at[-1, :, LEFT_VELOCITIES].set(
            discrete_velocities_prev[-2, :, LEFT_VELOCITIES])

        # 2 Macroscopic Velocities

        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(discrete_velocities_prev, density_prev)

        # Inflow ZOu;He

        macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, :].set(velocity_profile[0, 1:-1, :])
        density_prev = density_prev.at[0, :].set(
            (
                    get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES].T)
                    +
                    2 * get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES].T)
            ) / (
                    1 - macroscopic_velocities_prev[0, :, 0]
            )
        )

        # 4 Equilibrium

        equilibrium_discrete_velocities = get_equilibrium_velocities(macroscopic_velocities_prev, density_prev)

        # ZOu;he
        discrete_velocities_prev = discrete_velocities_prev.at[0, :, RIGHT_VELOCITIES].set(
            equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES])

        # Collision BGK

        discrete_velocities_post_collision = (
                discrete_velocities_prev
                -
                RELAXATION_OMEGA
                *
                (discrete_velocities_prev - equilibrium_discrete_velocities)
        )
        # Bounce Back no-slip On top and bottom

        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, 0,
                                                 LATTICE_INDICES[i]].set(
                discrete_velocities_prev[:, 0, OPPOSITE_LATTICE_INDICES[i]]
            )
            discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, -1,
                                                 LATTICE_INDICES[i]].set(
                discrete_velocities_prev[:, 0, OPPOSITE_LATTICE_INDICES[i]]

            )

        # Stream

        discrete_velocities_streamed = discrete_velocities_post_collision

        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:, :, i],
                        LATTICE_VELOCITIES[0, i],
                        axis=0,
                    ),
                    LATTICE_VELOCITIES[1, i],
                    axis=1
                )
            )
        return discrete_velocities_streamed

    # mesh

    x = jnp.arange(NX)
    y = jnp.arange(NY)

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    def run(discrete_velocities_prev):

        for iteration_idx in tqdm(range(N_ITERATIONS)):
            discrete_velocities_next = update(discrete_velocities_prev)
            discrete_velocities_prev = discrete_velocities_next

            if iteration_idx % PLOT_N_STEPS_TRUE == 0 and VISUALIZE and iteration_idx > SKIP_FIRST_N_ITERATIONS:
                density = get_density(discrete_velocities_next)
                macroscopic_velocities = (get_macroscopic_velocities(
                    discrete_velocities_next,
                    density
                ))

                velocity_magnitude = jnp.linalg.norm(
                    macroscopic_velocities,
                    axis=-1,
                    ord=2,
                )
                d_u__d_x, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
                d_v__d_x, d_v__d_y = jnp.gradient(macroscopic_velocities[..., 1])

                curl = (d_u__d_y - d_v__d_x)

                plt.subplot(211)
                plt.contourf(
                    X,
                    Y,
                    velocity_magnitude,
                    cmap=cmr.lavender,
                    levels=50,
                )
                plt.colorbar().set_label("Velocity Magnitude")

                plt.subplot(212)
                plt.contourf(
                    X,
                    Y,
                    curl,
                    levels=50,
                    vmin=-0.02,
                    vmax=0.02,
                    cmap=cmr.eclipse
                )
                plt.colorbar().set_label("Vorticity Magnitude")
                plt.draw()
                plt.pause(0.0001)
                plt.clf()

            if VISUALIZE:
                plt.show()




    run(get_equilibrium_velocities(velocity_profile, jnp.ones((NX, NY))))