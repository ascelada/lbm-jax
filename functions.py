import jax
import jax.numpy as jnp


@jax.jit
def collide(solver, discrete_velocities_prev, equilibrium_discrete_velocities):
    # Collision BGK
    discrete_velocities_post_collision = (
            discrete_velocities_prev
            - solver.RELAXATION_OMEGA * (discrete_velocities_prev - equilibrium_discrete_velocities)
    )

    # Bounce Back
    for i in range(solver.N_DISCRETE_VELOCITIES):
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
    # Calculate macroscopic velocities
    density_prev = solver.get_density(discrete_velocities_prev)
    macroscopic_velocities_prev = solver.get_macroscopic_velocities(discrete_velocities_prev, density_prev)
    macroscopic_velocities_prev = macroscopic_velocities_prev.at[:, :, 0].add(solver.ACCELERATION_MASK)

    # Calculate equilibrium
    equilibrium_discrete_velocities = solver.get_equilibrium_velocities(macroscopic_velocities_prev, density_prev)

    # Collision
    discrete_velocities_post_collision = collide(solver, discrete_velocities_prev, equilibrium_discrete_velocities)

    # Stream
    discrete_velocities_streamed = stream(solver, discrete_velocities_post_collision)

    return discrete_velocities_streamed
