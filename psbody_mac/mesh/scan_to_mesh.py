import numpy as np
import chumpy as ch
from .mesh_distance import MeshDistanceSquared
import scipy.sparse as sp

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

def GMOf(x, sigma):
    """Geman-McClure robustifier from sbody"""
    return SignedSqrt(x=GMOfInternal(x=x, sigma=sigma))

def construct_sampler(sampler, n_samples):
    """Construct a sampler for the scan points"""
    if isinstance(sampler, dict) and 'point2sample' in sampler:
        return sampler, sampler['point2sample'].shape[0]
    elif hasattr(sampler, 'v'):
        # If sampler is a mesh, use vertex sampling
        return {'point2sample': sp.eye(sampler.v.size, sampler.v.size)}, sampler.v.size // 3
    else:
        raise ValueError('Invalid sampler type')

class ScanToMesh(ch.Ch):
    """Compute distances between a scan and a mesh using the Geman-McClure robustifier"""
    dterms = 'mesh_verts'
    terms = 'scan', 'mesh_faces', 'rho'

    def __init__(self, scan, mesh_verts, mesh_faces, rho=None, normalize=True, signed=False):
        self.scan = scan
        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.rho = rho if rho is not None else (lambda x: x)
        self.normalize = normalize
        self.signed = signed
        
        # Construct sampler for scan points
        self.sampler, self.n_samples = construct_sampler(scan, scan.v.size // 3)

    def compute_r(self):
        # Create mesh distance term with exact geometric computations
        dist_squared = MeshDistanceSquared(reference_verts=self.mesh_verts,
                                         reference_faces=self.mesh_faces,
                                         sample_verts=self.scan.v,
                                         sampler=self.sampler)
        
        # Normalize if requested
        norm_const = np.sqrt(self.n_samples) if self.normalize else 1
        
        # Apply robustifier and normalization
        if self.signed:
            return SignedSqrt(self.rho(dist_squared)) / norm_const
        else:
            return ch.sqrt(self.rho(dist_squared)) / norm_const

    def compute_dr_wrt(self, wrt):
        if wrt is self.mesh_verts:
            # Create mesh distance term
            dist_squared = MeshDistanceSquared(reference_verts=self.mesh_verts,
                                             reference_faces=self.mesh_faces,
                                             sample_verts=self.scan.v,
                                             sampler=self.sampler)
            
            # Get derivative of distance term
            dist_dr = dist_squared.dr_wrt(wrt)
            
            # Apply chain rule with robustifier
            dr = dist_dr * self.rho.dr_wrt(dist_squared).reshape(-1, 1)
            
            # Apply normalization if requested
            if self.normalize:
                dr /= np.sqrt(self.n_samples)
            
            return dr
        
        return None 