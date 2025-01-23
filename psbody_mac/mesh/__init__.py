from .mesh_distance import MeshDistanceSquared, SignedSqrt, GMOfInternal
import numpy as np
import trimesh

class Mesh:
    """A basic mesh class that wraps trimesh functionality"""
    def __init__(self, v=None, f=None, filename=None):
        if filename is not None:
            mesh = trimesh.load(filename)
            self.v = mesh.vertices
            self.f = mesh.faces
        else:
            self.v = np.array(v) if v is not None else None
            self.f = np.array(f) if f is not None else None

    def write_obj(self, filename):
        mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        mesh.export(filename)
