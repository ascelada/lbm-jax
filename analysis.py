import h5py
import numpy as np


import h5py
import numpy as np

import h5py
import numpy as np

def compare_entrance_length(file_path, group_name, threshold, D, rho, mu):
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

        # Calculate mean entrance velocity to calculate Re
        u_mean_entrance = np.mean(velocity_matrix[0])
        print(u_mean_entrance)

        # Calculate Reynolds number at the entrance
        Re_entrance = rho * u_mean_entrance * D / mu

        print(Re_entrance)

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

