import statistics

import h5py
import numpy as np
from domain import label_islands,visualize_labeled_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from solver import  LBMFlowSolver
import jax.numpy as jnp
import cmasher as cmr

from scipy.interpolate import griddata

def compare_entrance_length(file_path, group_name, threshold,D, rho, mu):
    # Open the HDF5 file


    with h5py.File(file_path, 'r') as f:
        # Access the group containing the datasets
        group = f[group_name]

        # Get the names of all datasets in the group
        dataset_names = list(group.keys())


        # Sort the dataset names to ensure they are in order
        dataset_names.sort()

        # Get the name of the last dataset
        last_dataset_name = dataset_names[-1]

        # Get the velocity matrix from the group
        velocity_matrix = np.array(group[last_dataset_name])

        # Ensure the data is a float type
        velocity_matrix = velocity_matrix.astype(float)

        print(velocity_matrix)

        # Calculate mean entrance velocity to calculate Re
        u_mean_entrance = np.mean(velocity_matrix[0])

        # Calculate Reynolds number at the entrance
        Re_entrance = rho * u_mean_entrance * D / mu

        # Calculate analytical entrance length based on the entrance Reynolds number
        Le_analytical = 0.06 * Re_entrance * D



        # Iterate over the rows in the velocity_matrix to find the numerical entrance length
        for i, velocity_profile in enumerate(velocity_matrix):
            # If the standard deviation of the velocities is below the threshold, return the index (numerical entrance length)
            if np.std(velocity_profile) < threshold:
                # Calculate numerical entrance length (assuming axial increments of 1)
                Le_numerical = i * D
                return Le_numerical, Le_analytical

    # If no entrance length was found, return a sentinel value
    return -1, -1


def calculate_center(label_positions):
    centroids = {}

    for key, value in label_positions.items():
        x, y = zip(*value)
        centroids[key] = (np.mean(x), np.mean(y))

    return centroids

def calculate_node_forces(file_path):
    with h5py.File(file_path, 'r') as f:
        group = f["raw_data"]
        mask = f["mask_data"]['mask_data']
        labeled_matrix, label_positions = label_islands(mask)

        center_positions = calculate_center(label_positions)

        c = [[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]]

        forces_dict = {label: {'x':0, 'y':0} for label in label_positions.keys()}

        # Get the names of all datasets in the group
        dataset_names = list(group.keys())

        # Sort the dataset names to ensure they are in order
        dataset_names.sort()

        # Get the name of the last dataset
        last_dataset_name = dataset_names[-1]

        # Get the velocity matrix from the group
        raw_matrix = np.array(group[last_dataset_name])

        nx, ny, _ = raw_matrix.shape

        c = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
        c_inverse_index = [0, 3, 4, 1, 2, 7, 8, 5, 6]

        for x in range(nx):
            for y in range(ny):
                label = labeled_matrix[x][y]
                if label != 0:  # It's an object node
                    for i, ci in enumerate(c):
                        neighbor_x = (x + ci[0]) % nx
                        neighbor_y = (y + ci[1]) % ny
                        if labeled_matrix[neighbor_x][neighbor_y] == 0:  # It's a fluid node
                            #Get direction i
                            directional_vector = np.array(c[c_inverse_index[i]])

                            p_x,p_y = raw_matrix[neighbor_x][neighbor_y][c_inverse_index[i]] * directional_vector
                            p_x_out, p_y_out = raw_matrix[x][y][i] * directional_vector

                            forces_dict[label]['x'] += p_x - p_x_out
                            forces_dict[label]['y'] += p_y - p_y_out

        return forces_dict, center_positions, labeled_matrix


def print_forces(forces_dict, center_positions, labeled_matrix):

    centers_x = []
    centers_y = []
    total_forces_x = []
    total_forces_y =[]

    for key, value in center_positions.items():

        centers_x.append(value[0])
        centers_y.append(value[1])

    for key,value in forces_dict.items():
        total_forces_x.append(value['x'])
        total_forces_y.append(value['y'])
    centers_x = np.array(centers_x)
    centers_y = np.array(centers_y)
    total_forces_x = np.array(total_forces_x)
    total_forces_y = np.array(total_forces_y)
    nx, ny = np.array(labeled_matrix).shape


    plt.figure(figsize=(10, 10))
    plt.quiver(centers_x, centers_y, total_forces_x, total_forces_y, angles='xy', scale_units='xy', scale=0.06,
               color='blue')


    colors = ['#1E73DF'] + ['#9A9594'] * (max(forces_dict.keys()) + 1)
    cmap = mcolors.ListedColormap(colors)
    plt.imshow(labeled_matrix, alpha=1, cmap=cmap )  # Show islands as background
    plt.xlim(0, nx)
    plt.ylim(0, ny)
    plt.title('Force Field for Islands')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def calculate_statistics(forces_dict):
    x_values = [force["x"] for force in forces_dict.values()]
    y_values = [force["y"] for force in forces_dict.values()]

    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    std_dev_x = statistics.stdev(x_values)
    std_dev_y = statistics.stdev(y_values)
    print("Mean X:", mean_x)
    print("Mean Y:", mean_y)
    print("Standard Deviation X:", std_dev_x)
    print("Standard Deviation Y:", std_dev_y)

    z_scores_x = [(x - mean_x) / std_dev_x for x in x_values]

    # Calculate z-scores for y values
    z_scores_y = [(y - mean_y) / std_dev_y for y in y_values]

    # You can print these scores or analyze them further as needed
    print("Z-Scores for X values:", z_scores_x)
    print("Z-Scores for Y values:", z_scores_y)


# def plot_stream(file_path):
#     solver = LBMFlowSolver()
#     with h5py.File(file_path, 'r') as f:
#         group = f["raw_data"]
#         mask = jnp.array(f["mask_data"]['mask_data'])
#
#         dataset_names = list(group.keys())
#
#         # Sort the dataset names to ensure they are in order
#         dataset_names.sort()
#
#         # Get the name of the last dataset
#         last_dataset_name = dataset_names[-1]
#
#         # Get the velocity matrix from the group
#         raw_matrix = jnp.array(group[last_dataset_name])
#         density = solver.get_density(raw_matrix)
#         velocity_field = solver.get_macroscopic_velocities(solver, raw_matrix,density)
#         velocity_magnitude = jnp.linalg.norm(
#             velocity_field,
#             axis=-1,
#             ord=2,
#         )
#
#         # Creating a meshgrid for plotting
#         X,Y = velocity_magnitude.shape
#
#         plt.contourf(X, Y, velocity_field,
#                      levels=50, cmap=cmr.lavender)
#
#         plt.title("Enhanced Streamlines in LBM Simulation")
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.show()

# plot_stream('data.hdf5')
# print_forces(forces_dict, center_positions, labeled_matrix)
