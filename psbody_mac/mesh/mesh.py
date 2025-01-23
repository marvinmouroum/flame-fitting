import numpy as np
import trimesh
import scipy.sparse as sp

class Mesh(object):
    """A minimal implementation of psbody.mesh.Mesh using trimesh as backend"""
    def __init__(self, filename=None, v=None, f=None):
        if filename is not None:
            mesh = trimesh.load(filename)
            self.v = np.array(mesh.vertices, dtype=np.float64)
            self.f = np.array(mesh.faces, dtype=np.int32)
        else:
            self.v = np.array(v, dtype=np.float64) if v is not None else None
            self.f = np.array(f, dtype=np.int32) if f is not None else None
        
        self._compute_vertex_normals()
        self._compute_face_normals()
    
    def _compute_vertex_normals(self):
        """Compute vertex normals"""
        if self.v is None or self.f is None:
            self.vertex_normals = None
            return
        
        mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        self.vertex_normals = mesh.vertex_normals
    
    def _compute_face_normals(self):
        """Compute face normals"""
        if self.v is None or self.f is None:
            self.face_normals = None
            return
        
        mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        self.face_normals = mesh.face_normals
    
    def estimate_vertex_normals(self):
        """Re-estimate vertex normals"""
        self._compute_vertex_normals()
        return self.vertex_normals
    
    def estimate_face_normals(self):
        """Re-estimate face normals"""
        self._compute_face_normals()
        return self.face_normals
    
    def sample(self, num_samples, face_areas=None):
        """Sample points from mesh surface"""
        mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        
        # If face_areas not provided, compute them
        if face_areas is None:
            face_areas = mesh.area_faces
        
        # Normalize face areas to get probabilities
        face_probs = face_areas / np.sum(face_areas)
        
        # Sample face indices based on area
        face_indices = np.random.choice(len(self.f), size=num_samples, p=face_probs)
        
        # Generate random barycentric coordinates
        r1 = np.random.random(num_samples)
        r2 = np.random.random(num_samples)
        
        # Convert to barycentric coordinates
        u = 1 - np.sqrt(r1)
        v = r2 * np.sqrt(r1)
        w = 1 - u - v
        
        # Get vertices of sampled faces
        face_vertices = self.v[self.f[face_indices]]
        
        # Compute sample points using barycentric coordinates
        samples = (u[:, None] * face_vertices[:, 0] + 
                  v[:, None] * face_vertices[:, 1] + 
                  w[:, None] * face_vertices[:, 2])
        
        return samples
    
    def compute_aabb_tree(self):
        """Compute axis-aligned bounding box tree for the mesh"""
        self._trimesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        return self
    
    def compute_nearest_points(self, points):
        """Find nearest points on the mesh surface"""
        if not hasattr(self, '_trimesh'):
            self.compute_aabb_tree()
        
        closest_points, distances, face_indices = self._trimesh.nearest.on_surface(points)
        return closest_points, distances, face_indices
    
    def write(self, filename):
        """Write mesh to file"""
        mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        mesh.export(filename)
    
    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, value):
        if value is not None:
            self._v = np.array(value, dtype=np.float64)
        else:
            self._v = None
    
    @property
    def f(self):
        return self._f
    
    @f.setter
    def f(self, value):
        if value is not None:
            self._f = np.array(value, dtype=np.int32)
        else:
            self._f = None 