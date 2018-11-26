import os
import torch
from utils import makedir


NORMALIZE_INPUT = True     # Normalize input between min and max
MIN = -1
MAX = 0.992

LOAD_GEN = 'MSE_only'

SEED = 1    # None for random

MAIN_DIR = os.getcwd()
TEST_DIR = os.path.join(MAIN_DIR, 'Videos')
MODEL_DIR = os.path.join(MAIN_DIR, 'Models')

# directory for saved images
IMG_SAVE_DIR = makedir(os.path.join(MAIN_DIR, 'Predictions'))

CUDA = torch.cuda.is_available()

if CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# the number of history frames to give as input to the network
HIST_LEN = 8
# the number of predicted frames as output of the network
PRED_LEN = 1

# sequence length
SEQ_LEN = HIST_LEN + PRED_LEN


