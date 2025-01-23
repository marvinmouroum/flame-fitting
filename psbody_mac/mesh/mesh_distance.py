import numpy as np
import chumpy as ch
import scipy.sparse as sp
from .aabb_tree import AabbTree

class SignedSqrt(ch.Ch):
    """Signed square root function from sbody"""
    dterms = ('x',)
    terms = ()

    def compute_r(self):
        return np.sqrt(np.abs(self.x.r)) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            result = (.5 / np.sqrt(np.abs(self.x.r)))
            result = np.nan_to_num(result)
            result *= (self.x.r != 0).astype(np.uint32)
            return sp.spdiags(result.ravel(), [0], self.x.r.size, self.x.r.size)

class GMOfInternal(ch.Ch):
    """Geman-McClure robustifier internal computation from sbody"""
    dterms = 'x', 'sigma'

    def on_changed(self, which):
        if 'sigma' in which:
            assert(self.sigma.r > 0)

        if 'x' in which:
            self.squared_input = self.x.r ** 2.

    def compute_r(self):
        return (self.sigma.r ** 2 * (self.squared_input / (self.sigma.r ** 2 + self.squared_input))) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x and wrt is not self.sigma:
            return None

        squared_input = self.squared_input
        result = []
        if wrt is self.x:
            dx = self.sigma.r ** 2 / (self.sigma.r ** 2 + squared_input) - self.sigma.r ** 2 * (squared_input / (self.sigma.r ** 2 + squared_input) ** 2)
            dx = 2 * self.x.r * dx
            result.append(sp.spdiags((dx * np.sign(self.x.r)).ravel(), [0], self.x.r.size, self.x.r.size, format='csc'))
        if wrt is self.sigma:
            ds = 2 * self.sigma.r * (squared_input / (self.sigma.r ** 2 + squared_input)) - 2 * self.sigma.r ** 3 * (squared_input / (self.sigma.r ** 2 + squared_input) ** 2)
            result.append(sp.spdiags((ds * np.sign(self.x.r)).ravel(), [0], self.x.r.size, self.x.r.size, format='csc'))

        if len(result) == 1:
            return result[0]
        else:
            return np.sum(result).tocsc()

class MeshDistanceSquared(ch.Ch):
    """Compute squared distances between vertices and a mesh"""
    dterms = ('reference_verts', 'sample_verts')
    terms = ('reference_faces', 'sampler', 'signed')

    def __init__(self, reference_verts, reference_faces, sample_verts, sampler, signed=False):
        if not isinstance(reference_verts, ch.Ch):
            reference_verts = ch.array(reference_verts)
        if not isinstance(sample_verts, ch.Ch):
            sample_verts = ch.array(sample_verts)

        # Initialize terms
        terms = {}
        terms['reference_faces'] = reference_faces
        terms['sampler'] = sampler
        terms['signed'] = signed

        # Initialize dterms
        dterms = {}
        dterms['reference_verts'] = reference_verts
        dterms['sample_verts'] = sample_verts

        # Call parent constructor
        super(MeshDistanceSquared, self).__init__()

        # Set terms and dterms
        for k, v in terms.items():
            setattr(self, k, v)
        for k, v in dterms.items():
            setattr(self, k, v)

        self.tree = None

    def on_changed(self, which):
        if 'reference_verts' in which:
            ref_verts = self.reference_verts.r.reshape(-1, 3)
            self.tree = AabbTree(ref_verts, self.reference_faces)

    def compute_r(self):
        if self.tree is None:
            ref_verts = self.reference_verts.r.reshape(-1, 3)
            self.tree = AabbTree(ref_verts, self.reference_faces)

        # Compute distances from sample points to reference mesh
        sample_points = self.sample_verts.r.reshape(-1, 3)
        if self.sampler is not None:
            sample_points = self.sampler.dot(sample_points.reshape(-1)).reshape(-1, 3)

        distances = np.zeros(len(sample_points))
        for i, point in enumerate(sample_points):
            closest_point, face_idx = self.tree.nearest(point)
            dist = np.sum((point - closest_point) ** 2)
            if self.signed:
                # Compute normal of the closest face
                face = self.reference_faces[face_idx]
                v1, v2, v3 = self.reference_verts.r.reshape(-1, 3)[face]
                normal = np.cross(v2 - v1, v3 - v1)
                normal = normal / np.linalg.norm(normal)
                # Sign the distance based on which side of the face the point is on
                sign = np.sign(np.dot(point - closest_point, normal))
                dist *= sign
            distances[i] = dist

        return distances

    def compute_dr_wrt(self, wrt):
        if wrt not in (self.reference_verts, self.sample_verts):
            return None

        sample_points = self.sample_verts.r.reshape(-1, 3)
        if self.sampler is not None:
            sample_points = self.sampler.dot(sample_points.reshape(-1)).reshape(-1, 3)

        if wrt is self.reference_verts:
            # Derivative with respect to reference vertices
            J = np.zeros((len(sample_points), self.reference_verts.r.size))
            for i, point in enumerate(sample_points):
                closest_point, face_idx = self.tree.nearest(point)
                face = self.reference_faces[face_idx]
                for j, vertex_idx in enumerate(face):
                    # Simple approximation: distribute gradient equally among vertices of closest face
                    J[i, vertex_idx * 3:(vertex_idx + 1) * 3] = 2 * (closest_point - point) / 3

            if self.signed:
                # Add derivative of sign computation
                for i, point in enumerate(sample_points):
                    closest_point, face_idx = self.tree.nearest(point)
                    face = self.reference_faces[face_idx]
                    v1, v2, v3 = self.reference_verts.r.reshape(-1, 3)[face]
                    normal = np.cross(v2 - v1, v3 - v1)
                    normal = normal / np.linalg.norm(normal)
                    sign = np.sign(np.dot(point - closest_point, normal))
                    for vertex_idx in face:
                        J[i, vertex_idx * 3:(vertex_idx + 1) * 3] *= sign

            return sp.csc_matrix(J)

        else:  # wrt is self.sample_verts
            # Derivative with respect to sample vertices
            J = np.zeros((len(sample_points), self.sample_verts.r.size))
            for i, point in enumerate(sample_points):
                closest_point, _ = self.tree.nearest(point)
                J[i, i * 3:(i + 1) * 3] = 2 * (point - closest_point)

            if self.signed:
                # Add derivative of sign computation
                for i, point in enumerate(sample_points):
                    closest_point, face_idx = self.tree.nearest(point)
                    face = self.reference_faces[face_idx]
                    v1, v2, v3 = self.reference_verts.r.reshape(-1, 3)[face]
                    normal = np.cross(v2 - v1, v3 - v1)
                    normal = normal / np.linalg.norm(normal)
                    sign = np.sign(np.dot(point - closest_point, normal))
                    J[i, i * 3:(i + 1) * 3] *= sign

            if self.sampler is not None:
                J = self.sampler.T.dot(J)

            return sp.csc_matrix(J) 