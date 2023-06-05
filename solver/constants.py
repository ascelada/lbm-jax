import  jax.numpy as jnp

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