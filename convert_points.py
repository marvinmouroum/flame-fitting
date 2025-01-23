import numpy as np
from fitting.landmarks import load_picked_points

# Load .pp file
landmarks = load_picked_points('data/MarvinKopf_lmks.pp')

# Save as .npy
np.save('data/MarvinKopf_lmks.npy', landmarks)