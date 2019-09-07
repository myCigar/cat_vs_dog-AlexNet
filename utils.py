import os
import torch

def save_model(state, directory='./checkpoints', filename=None):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    pkl_filename = os.path.join(directory, filename)
    torch.save(state, pkl_filename)
    print('Save "{:}" in {:} successful'.format(pkl_filename, directory))