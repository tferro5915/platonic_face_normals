"""Platonic Face Normals: Visualize rotating platonic solid, showing all face normal vectors, based on one known face normal vector.
"""

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from itertools import product, combinations

eps = 0.00001

class platonic_enum(Enum):
    """Used for selecting which platonic solid to use. 
    """
    tetrahedron = 4
    cube = 6
    octahedron = 8
    dodecahedron = 12
    icosahedron = 20


class platonic:
    """Creates a platonic solid to reorient based on one face's normal vector. 
    """

    def __init__(self, shape: platonic_enum):
        """Initialize the class instance

        :param shape: Defines the shape of the platonic solid to be used
        :type shape: platonic_enum
        """
        self.normals = standard_normals[shape]
        self.edges = edges

    def orient_to(self, vector: np.array):
        """Re-orient the shape based on one face's normal vector.

        :param vector: A face's normal vector
        :type vector: np.array
        """
        self.vector = vector / np.linalg.norm(vector)
        rot = rotation_matrix_from_vectors(self.vector, self.normals[0])
        self.normals = [normal.dot(rot) for normal in self.normals]
        self.edges = [{'s': edge['s'].dot(rot), 'e': edge['e'].dot(rot)} for edge in self.edges]

    def plot(self, show_cube:bool = False):
        """Show a plot of the platonic solid faces' normal vectors. Optionally shows a cube since many have vectors related to a cube. 

        :param show_cube: Also show a cube, defaults to False
        :type show_cube: bool, optional
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot initial vector
        ax.quiver(0, 0, 0, self.vector[0], self.vector[1], self.vector[2], length=np.linalg.norm(self.vector)*1.25, normalize=True, color="r")

        # Draw normal vectors
        for normal in self.normals:
            ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], length=np.linalg.norm(normal), normalize=True, color="b")

        # draw cube
        if show_cube:
            for edge in self.edges:
                ax.plot3D(*zip(edge['s'], edge['e']), color="g")

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def rotation_matrix_from_vectors(vec1: np.array, vec2: np.array) -> np.array:
    """Find the rotation matrix that aligns vec1 to vec2
    
    source: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    :param vec1: A 3d "source" vector
    :type vec1: np.array
    :param vec2: A 3d "destination" vector
    :type vec2: np.array
    :return: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    :rtype: np.array
    """
    
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + eps))
    return rotation_matrix


# Icosahedron values
phi = (1 + 5 ** 0.5) / 2  # 1.6180
I1 = 0.1876  # 0.1876  # TODO find formula values for these
I2 = 0.6071  # 0.6071
I3 = 0.4911  # 0.4911
I4 = 0.9822  # 0.9822
I5 = 0.3035 # 0.3035
I6 = 0.7946  # 0.7946
I7 = 1/(3 ** 0.5)  # 0.5774
I8 = I7 * phi  # 0.9342
I9 = I7 * (phi - 1)  # 0.3568


# Make initial vectors
standard_normals = {
    platonic_enum.tetrahedron: [
        np.array([1, -1, -1]),
        np.array([-1, 1, -1]),
        np.array([-1, -1, 1]),
        np.array([1, 1, 1])],
    
    platonic_enum.cube: [
        np.array([1, 0, 0]),
        np.array([-1, 0, 0]),
        np.array([0, 1, 0]),
        
        np.array([0, -1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, -1])],
    
    platonic_enum.octahedron: [
        np.array([1, 1, 1]),
        np.array([-1, -1, -1]),
        np.array([-1, 1, 1]),
        np.array([1, -1, -1]),
        
        np.array([1, -1, 1]),
        np.array([-1, 1, -1]),
        np.array([1, 1, -1]),
        np.array([-1, -1, 1])],
    
    platonic_enum.dodecahedron: [
        np.array([1, 1, 0]),
        np.array([-1, -1, 0]),
        np.array([1, -1, 0]),
        np.array([-1, 1, 0]),

        np.array([0, 1, 1]),
        np.array([0, -1, -1]),
        np.array([0, -1, 1]),
        np.array([0, 1, -1]),

        np.array([1, 0, 1]),
        np.array([-1, 0, -1]),
        np.array([-1, 0, 1]),
        np.array([1, 0, -1])],
    
    platonic_enum.icosahedron: [
        np.array([I1, -I6, I7]),
        np.array([I2, -I6, 0]),
        np.array([-I3, -I6, I9]),
        np.array([-I3, -I6, -I9]),
        
        np.array([I1, -I6, -I7]),
        np.array([I4, -I1, 0]),
        np.array([I5, -I1, I8]),
        np.array([-I6, -I1, I7]),
        
        np.array([-I6, -I1, -I7]),
        np.array([I5, -I1, -I8]),
        np.array([I6, I1, I7]),
        np.array([-I5, I1, I8]),
        
        np.array([-I4, I1, 0]),
        np.array([-I5, I1, -I8]),
        np.array([I6, I1, -I7]),
        np.array([I3, I6, I9]),
        
        np.array([-I1, I6, I7]),
        np.array([-I2, I6, 0]),
        np.array([-I1, I6, -I7]),
        np.array([I3, I6, -I9])]
    } 


# Convert to unit vectors
for (key, normals) in standard_normals.items():
    standard_normals[key] = [normal / np.linalg.norm(normal) for normal in normals]


# Outline cube make
r = [-1, 1]
edges = []
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        s = np.array(s)
        e = np.array(e)
        s = s / np.linalg.norm(s)
        e = e / np.linalg.norm(e)
        edges.append({'s': s, 'e': e})


if __name__ == '__main__':
    p = platonic(platonic_enum.octahedron)
    p.orient_to(np.array([1, 1, 1]))
    p.plot(True)
