# LBM-JAX

## Description
This project, named "LBM-JAX", is a computational fluid dynamics (CFD) solver based on the Lattice Boltzmann Method (LBM), implemented in Python using JAX for high-performance, differentiable computing. 

The `main.py` script initializes the simulation, updating the JAX configuration for enhanced precision. It loads a domain matrix, sets up the LBM solver, and provides key simulation parameters like Reynolds number and porosity. Additionally, it includes visualization functionality for the labeled matrix.

The `solver.py` file contains the `LBMFlowSolver` class, which is central to the simulation. This class integrates various libraries like `cmasher`, `flax`, `h5py`, and `matplotlib` for data handling, neural network structuring, file operations, and data visualization, respectively. It also defines the LBM solver's physical parameters and functions for animation and visualization of the flow simulation.

In `functions.py`, critical computational functions for the LBM solver are defined. It includes the implementation of the collision step in the LBM using the Bhatnagar-Gross-Krook (BGK) model, optimized with JAX's `@jax.jit` decorator for performance. This script ensures the efficient calculation of post-collision discrete velocities, integral to the LBM's computational process.

## Features
- Lattice Boltzmann Method (LBM) based fluid flow simulation
- Implemented using JAX for high performance and differentiability
- Visualization of fluid flow and domain matrices
- Customizable simulation parameters
