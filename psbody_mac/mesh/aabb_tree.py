import numpy as np
import trimesh

class AabbTree:
    """A wrapper around trimesh's proximity query functionality"""
    def __init__(self, vertices, faces):
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.mesh.vertices.flags.writeable = False
        self.mesh.faces.flags.writeable = False

    def nearest(self, points, nearest_part=False):
        """Find the nearest points on the mesh to the query points"""
        closest_points, distances, face_indices = trimesh.proximity.closest_point(self.mesh, points)
        
        if not nearest_part:
            return closest_points, distances, face_indices
        
        # For each point, determine if it's closest to a vertex, edge, or face
        parts = np.zeros(len(points), dtype=np.uint64)
        
        for i, (point, face_idx) in enumerate(zip(points, face_indices)):
            face = self.mesh.faces[face_idx]
            v0, v1, v2 = self.mesh.vertices[face]
            
            # Compute barycentric coordinates
            v0_to_p = point - v0
            v0_to_v1 = v1 - v0
            v0_to_v2 = v2 - v0
            
            # Compute area coordinates
            area = np.cross(v0_to_v1, v0_to_v2)
            area_norm = np.linalg.norm(area)
            
            if area_norm > 1e-10:
                area = area / area_norm
                
                # Project point onto triangle plane
                proj = point - np.dot(v0_to_p, area) * area
                
                # Compute barycentric coordinates
                v = np.cross(v0_to_v1, proj - v0)
                w = np.cross(proj - v0, v0_to_v2)
                denom = np.dot(area, np.cross(v0_to_v1, v0_to_v2))
                
                if abs(denom) > 1e-10:
                    b1 = np.dot(area, v) / denom
                    b2 = np.dot(area, w) / denom
                    b0 = 1 - b1 - b2
                    
                    # Determine which part is closest
                    if b0 >= 0 and b1 >= 0 and b2 >= 0:
                        parts[i] = 0  # Face
                    elif b1 < 0:
                        parts[i] = 3  # Edge between v2 and v0
                    elif b2 < 0:
                        parts[i] = 1  # Edge between v0 and v1
                    else:
                        parts[i] = 2  # Edge between v1 and v2
                else:
                    # Degenerate triangle, treat as vertex
                    parts[i] = 4  # First vertex
            else:
                # Degenerate triangle, treat as vertex
                parts[i] = 4  # First vertex
        
        return closest_points, parts, face_indices 