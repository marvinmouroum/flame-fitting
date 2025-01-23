import numpy as np
import trimesh

def aabbtree_compute(v, f):
    """Create an AabbTree instance"""
    return AabbTree(v, f)

def aabbtree_nearest(tree_handle, points):
    """Wrapper for nearest function to match original C++ interface"""
    closest_points, distances, face_indices = tree_handle.nearest(points)
    # Create a part array (0=face, 1-3=edge, 4-6=vertex)
    parts = np.zeros_like(face_indices)  # All faces for now
    return face_indices, parts, closest_points

class AabbTree(object):
    """A minimal implementation of AABB tree using trimesh's built-in spatial indexing"""
    def __init__(self, v, f):
        self.mesh = trimesh.Trimesh(vertices=v, faces=f)
        # Force building of the spatial index
        _ = self.mesh.nearest.on_surface([[0, 0, 0]])
    
    def nearest(self, points, nearest_part=False):
        """Find nearest points on the mesh surface"""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        closest_points, distances, face_indices = self.mesh.nearest.on_surface(points)
        
        # nearest_part is now treated as a flag rather than an output parameter
        if nearest_part:
            # Create a part array (0=face, 1-3=edge, 4-6=vertex)
            parts = np.zeros_like(face_indices)  # All faces for now
            return closest_points, distances, face_indices, parts
        
        return closest_points, distances, face_indices
    
    def nearest_alongnormal(self, points, normals):
        """Find nearest points along given normal directions"""
        points = np.asarray(points)
        normals = np.asarray(normals)
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
            normals = normals.reshape(1, -1)
            
        # Normalize normals
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Use ray casting to find intersections along normals
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=points,
            ray_directions=normals
        )
        
        # Calculate distances
        if len(locations) > 0:
            distances = np.linalg.norm(locations - points[index_ray], axis=1)
        else:
            distances = np.array([])
            
        return distances, index_tri, locations
    
    def all_intersections(self, ray_origins, ray_directions=None):
        """Find all intersections with rays"""
        if ray_directions is None:
            # If only one parameter is provided, assume it's a Ray object
            ray_directions = ray_origins.directions
            ray_origins = ray_origins.origins
        
        ray_origins = np.asarray(ray_origins)
        ray_directions = np.asarray(ray_directions)
        
        if ray_origins.ndim == 1:
            ray_origins = ray_origins.reshape(1, -1)
            ray_directions = ray_directions.reshape(1, -1)
        
        # Normalize ray directions
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)
        
        # Find intersections
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )
        
        return locations, index_ray, index_tri