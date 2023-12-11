import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import jax
from numba import njit
from numba import cuda
import jax.numpy as jnp

def generate_lattice_spheres(r, nx, ny, bed_value, volume_fraction=1):
    shape = [nx,int(bed_value*ny)]

    o = 1.25 * r
    s = 3.75 * r

    im = ps.generators.rsa(im_or_shape=shape,volume_fraction=volume_fraction, r=r, clearance=2)
    NX,NY = shape
    matrix = np.array(im)
    porosity = ((NX*NY)-np.sum(im))/(NX*NY)
    print(porosity)
    print(matrix.shape)
    teste = np.full([nx, int((1-bed_value)*ny)], False)

    matrix = np.hstack((matrix,teste))
    return matrix

def label_islands(matrix):
    rows, cols = len(matrix), len(matrix[0])
    labels = [[0] * cols for _ in range(rows)]
    label_positions = {}
    current_label = 0


    def dfs(i, j, label):
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            if (0 <= x < rows and 0 <= y < cols and
                    matrix[x][y] and labels[x][y] == 0):
                labels[x][y] = label
                label_positions[label].append((x, y))
                for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    stack.append((nx, ny))
    print(rows)
    for i in range(rows):
        print(i)
        for j in range(cols):
            if matrix[i][j] and labels[i][j] == 0:
                current_label += 1  # Increment label for a new island
                label_positions[current_label] = []
                dfs(i, j, current_label)

    return labels, label_positions



def visualize_labeled_matrix(matrix):
    plt.imshow(matrix, cmap='rainbow', interpolation='none')
    plt.colorbar(label='Label')
    plt.rcParams['axes.facecolor'] = 'white'

    # Adding grid lines
    plt.grid(which='both', linestyle='-', linewidth=1)

    plt.title('Labeled Islands')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    plt.show()


