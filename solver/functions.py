import jax.numpy as jnp
import constants as cts


def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)  # last axis
    return density
def update(discrete_velocities_prev, LBMSolver, velocity_profile):
    # 1 Prescribe outflow on right boundary null-gradient
    discrete_velocities_prev = discrete_velocities_prev.at[-1, :, cts.LEFT_VELOCITIES].set(
        discrete_velocities_prev[-2, :, cts.LEFT_VELOCITIES])

    # 2 Macroscopic Velocities

    density_prev = get_density(discrete_velocities_prev)
    macroscopic_velocities_prev = get_macroscopic_velocities(discrete_velocities_prev, density_prev)

    # Inflow Zou/He

    macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, :].set(velocity_profile[0, 1:-1, :])
    density_prev = density_prev.at[0, :].set(
        (
                get_density(discrete_velocities_prev[0, :, cts.PURE_VERTICAL_VELOCITIES].T)
                +
                2 * get_density(discrete_velocities_prev[0, :, cts.LEFT_VELOCITIES].T)
        ) / (
                1 - macroscopic_velocities_prev[0, :, 0]
        )
    )

    # 4 Equilibrium

    equilibrium_discrete_velocities = get_equilibrium_velocities(macroscopic_velocities_prev, density_prev)

    # Zou/he
    discrete_velocities_prev = discrete_velocities_prev.at[0, :, cts.RIGHT_VELOCITIES].set(
        equilibrium_discrete_velocities[0, :, cts.RIGHT_VELOCITIES])

    # Collision BGK

    discrete_velocities_post_collision = (
            discrete_velocities_prev
            -
            RELAXATION_OMEGA
            *
            (discrete_velocities_prev - equilibrium_discrete_velocities)
    )
    # Bounce Back no-slip On top and bottom

    for i in range(LBMSolver.n_discrete_velocities):
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, 0,
                                             cts.LATTICE_INDICES[i]].set(
            discrete_velocities_prev[:, 0, cts.OPPOSITE_LATTICE_INDICES[i]]
        )
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, -1,
                                             cts.LATTICE_INDICES[i]].set(
            discrete_velocities_prev[:, 0, cts.OPPOSITE_LATTICE_INDICES[i]]

        )

    # Stream

    discrete_velocities_streamed = discrete_velocities_post_collision

    for i in range(LBMSolver.n_discrete_velocities):
        discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
            jnp.roll(
                jnp.roll(
                    discrete_velocities_post_collision[:, :, i],
                    cts.LATTICE_VELOCITIES[0, i],
                    axis=0,
                ),
                cts.LATTICE_VELOCITIES[1, i],
                axis=1
            )
        )
    return discrete_velocities_streamed