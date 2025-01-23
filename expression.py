import numpy as np
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir
from os.path import join
import pickle

def create_expression(model_path, output_dir, expression_params=None):
    """
    Load a FLAME model and modify its expression
    
    Args:
        model_path: Path to the fitted model parameters
        output_dir: Directory to save the new expressiond
        expression_params: Optional array of expression parameters (100 values)
    """
    # Load the FLAME model
    model = load_model('./models/generic_model.pkl')
    
    # Load the fitted parameters
    with open(model_path, 'rb') as f:
        fitted_params = pickle.load(f)
    
    # Set the fitted shape and pose parameters
    model.betas[:300] = fitted_params['betas'][:300]  # Shape parameters
    model.pose[:] = fitted_params['pose']  # Pose parameters
    model.trans[:] = fitted_params['trans']  # Translation parameters
    
    # Set new expression parameters
    if expression_params is None:
        # Example: Create a smile expression
        # These values are approximate - you may need to adjust them
        expression = np.zeros(100)
        expression[0] = 0.5    # Lip corner puller (smile)
        expression[1] = 0.2    # Cheek raiser
        expression[4] = 0.1    # Lip part
    else:
        expression = expression_params
    
    # Apply the expression parameters (indices 300-399)
    model.betas[300:400] = expression
    
    # Create output directory if it doesn't exist
    safe_mkdir(output_dir)

    filename = model_path.split('/')[-1]
    
    # Save the new mesh
    output_path = join(output_dir, filename + '_expression.obj')
    write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=output_path)
    print('Output mesh saved to:', output_path)

if __name__ == '__main__':
    # Path to your fitted model parameters (from the previous fitting)
    fitted_model_path = './output/MarvinKopf_params.pkl'
    output_dir = './output'
    
    create_expression(fitted_model_path, output_dir)
