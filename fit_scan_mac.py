'''
Mac-compatible version of the FLAME fitting script
'''

import numpy as np
import chumpy as ch
import trimesh
from psbody_mac.mesh import Mesh, MeshDistanceSquared, SignedSqrt, GMOfInternal
import os
import pickle
import scipy.sparse as sp

class Model:
    """A wrapper for the FLAME model"""
    def __init__(self, model_dict):
        # Load model data
        self.v_template = ch.array(model_dict['v_template'].reshape(-1))  # Flatten to 1D
        self.f = model_dict['f']
        self.shapedirs = ch.array(model_dict['shapedirs'].reshape(-1, 300))  # Reshape to 2D
        self.posedirs = ch.array(model_dict['posedirs'].reshape(-1, 12))  # Reshape to 2D
        
        # FLAME uses blend shapes for expressions
        self.bs_type = model_dict['bs_type']
        self.bs_style = model_dict['bs_style']
        
        # Initialize vertices
        self.v = None

def create_sampler(vertices, faces):
    """Create a sampler that uses all vertices"""
    num_vertices = len(vertices)
    indices = np.arange(num_vertices)
    values = np.ones(num_vertices)
    point2sample = sp.csc_matrix((values, (indices, indices)), shape=(num_vertices, num_vertices))
    return point2sample.toarray()  # Convert to numpy array

def load_model(fname_or_dict, model_type='male'):
    if isinstance(fname_or_dict, str):
        if not os.path.exists(fname_or_dict):
            raise ValueError('Path %s does not exist!' % fname_or_dict)
        with open(fname_or_dict, 'rb') as f:
            dd = pickle.load(f, encoding='latin1')
    else:
        dd = fname_or_dict

    model_type = model_type.lower()
    if model_type == 'male':
        fname = 'models/male_model.pkl'
    elif model_type == 'female':
        fname = 'models/female_model.pkl'
    else:
        raise ValueError('Unknown model type: %s' % model_type)

    if not os.path.exists(fname):
        raise ValueError('Path %s does not exist!' % fname)
    with open(fname, 'rb') as f:
        model_dict = pickle.load(f, encoding='latin1')
        return Model(model_dict)

def fit_scan(scan_path, model_type='male', weights=None):
    # Load the scan
    scan = trimesh.load(scan_path)
    scan_v = scan.vertices
    scan_f = scan.faces

    # Load the FLAME model
    model = load_model(None, model_type=model_type)

    # Set up the optimization
    # Initialize parameters
    pose = ch.zeros(12)  # 3 jaw + 9 neck
    shape = ch.zeros(300)  # Shape parameters
    trans = ch.zeros(3)  # Translation

    # Set up the model
    # FLAME uses shape blend shapes only
    model.v = model.v_template + model.shapedirs.dot(shape) + model.posedirs.dot(pose)
    
    # Broadcast translation to all vertices
    num_vertices = model.v_template.size // 3
    trans_broadcast = ch.tile(trans, num_vertices)
    model.v = model.v + trans_broadcast

    # Create a sampler for the scan
    sampler = create_sampler(scan_v, scan_f)

    # Convert scan vertices to chumpy array
    scan_v_ch = ch.array(scan_v.reshape(-1))

    # Set up the objective function terms
    # Scan-to-mesh distance term with Geman-McClure robustifier
    scan_to_mesh_dist = MeshDistanceSquared(model.v, model.f, scan_v_ch, sampler, signed=True)
    sigma = ch.array([0.1])  # Robustifier parameter
    scan_to_mesh_error = GMOfInternal(SignedSqrt(scan_to_mesh_dist), sigma)

    # Regularization terms
    shape_reg = weights.get('shape', 1.0) * shape.dot(shape)
    pose_reg = weights.get('pose', 1.0) * pose.dot(pose)

    # Total objective
    objectives = {
        'scan_to_mesh': scan_to_mesh_error,
        'shape_reg': shape_reg,
        'pose_reg': pose_reg
    }

    # Optimize
    steps = [
        # First optimize translation only
        {'maxiter': 50, 'free_vars': [trans]},
        # Then optimize shape
        {'maxiter': 50, 'free_vars': [trans, shape]},
        # Finally optimize everything including pose
        {'maxiter': 100, 'free_vars': [trans, shape, pose]}
    ]

    for step in steps:
        ch.minimize(
            objectives,
            x0=step['free_vars'],
            method='dogleg',
            options={'maxiter': step['maxiter']},
            callback=None
        )

    # Return the fitted model
    return {
        'v': model.v.r.reshape(-1, 3),
        'f': model.f,
        'pose': pose.r,
        'shape': shape.r,
        'trans': trans.r
    }

if __name__ == '__main__':
    # Example usage
    scan_path = 'data/scan.obj'
    weights = {
        'shape': 0.1,  # Reduced shape regularization
        'pose': 0.1    # Reduced pose regularization
    }
    result = fit_scan(scan_path, model_type='male', weights=weights)
    
    # Save the result
    output_mesh = Mesh(v=result['v'], f=result['f'])
    output_mesh.write_obj('output_fit.obj') 