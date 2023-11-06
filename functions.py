import jax

import jax.numpy as jnp
@jax.jit
def get_force(discrete_velocities):
  return jnp.sum(
                 (LATTICE_VELOCITIES.T[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *
                  discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_IN] +
                 (LATTICE_VELOCITIES.T[OPPOSITE_LATTICE_INDICES][jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *
                  discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_OUT],
                 axis = 0)

@jax.jit
def collide(solver, discrete_velocities_prev, equilibrium_discrete_velocities):
    # Collision BGK
    discrete_velocities_post_collision = (
            discrete_velocities_prev
            - solver.RELAXATION_OMEGA * (discrete_velocities_prev - equilibrium_discrete_velocities)
    )

    # Bounce Back no-slip On top and bottom
    for i in range(solver.N_DISCRETE_VELOCITIES):
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, 0,
                                             solver.LATTICE_INDICES[i]].set(
            discrete_velocities_prev[:, 0, solver.OPPOSITE_LATTICE_INDICES[i]]
        )
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, -1,
                                             solver.LATTICE_INDICES[i]].set(
            discrete_velocities_prev[:, 0, solver.OPPOSITE_LATTICE_INDICES[i]]
        )

        if solver.isPorous:
            discrete_velocities_post_collision = \
                discrete_velocities_post_collision.at[solver.mask, solver.LATTICE_INDICES[i]].set(
                    discrete_velocities_prev[solver.mask, solver.OPPOSITE_LATTICE_INDICES[i]]
                )

    return discrete_velocities_post_collision


@jax.jit
def stream(solver, discrete_velocities_post_collision):
    # Stream
    discrete_velocities_streamed = discrete_velocities_post_collision
    for i in range(solver.N_DISCRETE_VELOCITIES):
        discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
            jnp.roll(
                jnp.roll(
                    discrete_velocities_post_collision[:, :, i],
                    solver.LATTICE_VELOCITIES[0, i],
                    axis=0,
                ),
                solver.LATTICE_VELOCITIES[1, i],
                axis=1
            )
        )
    return discrete_velocities_streamed


@jax.jit
def update(solver, discrete_velocities_prev):
    # Prescribe outflow on right boundary null-gradient
    discrete_velocities_prev = discrete_velocities_prev.at[-1, :, solver.LEFT_VELOCITIES].set(
        discrete_velocities_prev[-2, :, solver.LEFT_VELOCITIES]
    )

    # Calculate macroscopic velocities
    density_prev = solver.get_density(discrete_velocities_prev)
    macroscopic_velocities_prev = solver.get_macroscopic_velocities(discrete_velocities_prev, density_prev)

    # Inflow Zou/He
    macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, :].set(solver.velocity_profile[0, 1:-1, :])
    density_prev = density_prev.at[0, :].set(
        (
                solver.get_density(discrete_velocities_prev[0, :, solver.PURE_VERTICAL_VELOCITIES].T)
                + 2 * solver.get_density(discrete_velocities_prev[0, :, solver.LEFT_VELOCITIES].T)
        ) / (1 - macroscopic_velocities_prev[0, :, 0])
    )

    # Calculate equilibrium
    equilibrium_discrete_velocities = solver.get_equilibrium_velocities(macroscopic_velocities_prev, density_prev)

    # Zou/he
    discrete_velocities_prev = discrete_velocities_prev.at[0, :, solver.RIGHT_VELOCITIES].set(
        equilibrium_discrete_velocities[0, :, solver.RIGHT_VELOCITIES]
    )

    # Collision
    discrete_velocities_post_collision = collide(solver, discrete_velocities_prev, equilibrium_discrete_velocities)

    # Stream
    discrete_velocities_streamed = stream(solver,discrete_velocities_post_collision)

    return discrete_velocities_streamed
