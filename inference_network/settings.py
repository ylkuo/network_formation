import networkx as nx
import random as rd
import numpy as np

import torch

rd.seed(0)
np.random.seed(0)

use_gpu = torch.cuda.is_available() and False
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

# Settings for the generative model
gen_model_params = dict()
gen_model_params['size'] = 10
gen_model_params['feature_length'] = gen_model_params['size']
gen_model_params['theta_range'] = (0, 6)

# Settings for the inference network
n_features = gen_model_params['size']
hidden_dim = 32
rnn_depth = 2
embedding_dim = 32
prior_mean = 3
prior_stddev = 1

# Settings for training
n_train = 200
n_validation = 20
size_per_theta_train = 20
n_epochs = 500
lr = 0.0001
weight_decay=0.00005
is_train = False

# Settings for saving/loading models
model_prefix = 'model/'
model_name = 'formation.model'
checkpoint_range = 50

# Settings for evaluation plots
save_fig = True
show_fig = True
estimator_type = 'posterior_mean'
n_eval_thetas = 40
n_eval_theta_samples = 10
