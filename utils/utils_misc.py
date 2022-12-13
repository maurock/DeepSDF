from random import random
import numpy as np
import torch
import random

# set seeds for consistency
def set_seeds(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	random.seed(seed)