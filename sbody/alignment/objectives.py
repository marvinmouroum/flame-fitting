#__all__ = ['landmark_function', 'scan_to_mesh_squared_function',
#           'mesh_to_scan_squared_function', 'scan_to_mesh_function',
#           'mesh_to_scan_function', 'full_mesh_to_scan_function', 'vecdiff_function']

from scipy.sparse import csc_matrix
import numpy as np
import scipy as sp
import scipy.sparse as sp
import sbody.alignment.mesh_distance.mesh_distance as mesh_distance
import random
import copy

from sbody.matlab import *

def co3(x):
    return matlab.bsxfun(np.add, row(np.arange(3)), col(3 * (x)))


def triangle_area(v, f):
    return np.sqrt(np.sum(np.cross(v[f[:, 1], :] - v[f[:, 0], :], v[f[:, 2], :] - v[f[:, 0], :]) ** 2, axis=1)) / 2


def sample_categorical(samples, dist):
    a = np.random.multinomial(samples, dist)
    b = np.zeros(int(samples), dtype=int)
    upper = np.cumsum(a)
    lower = upper - a
    for value in range(len(a)):
        b[lower[value]: upper[value]] = value
    np.random.shuffle(b)
    return b


def sample_from_mesh(mesh, sample_type='vertices'):
    if sample_type == 'vertices':
        sample_spec = {'point2sample': sp.eye(mesh.v.size, mesh.v.size)}  # Direct use of sp (scipy.sparse)
        return sample_spec
    else:
        raise Exception('Unknown sample type: %s' % (sample_type,))
