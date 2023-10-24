import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import jax

import jax.numpy as jnp

def generate_lattice_spheres(r, nx, ny, bed_value):
    shape = [nx,int(bed_value*ny)]

    o = 1.25 * r
    s = 3.75 * r

    im = ~ps.generators.lattice_spheres(shape=shape, r=r, spacing=s, offset=o, lattice='tri')
    matrix = jnp.array(im)
    print(matrix.shape)
    teste = jnp.full([nx, int((1-bed_value)*ny)], False)

    matrix = jnp.hstack((matrix, teste))
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

    for i in range(rows):
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

