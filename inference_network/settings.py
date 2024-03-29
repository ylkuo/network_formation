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
gen_model_params['input_type'] = 'clustering'  # 'degree_sequence'  # degree_sequence, adjacencies
gen_model_params['space_between_observations'] = 6  # take every 6 points to generate the time series

# Settings for the inference network
n_nodes = gen_model_params['size']
n_features = gen_model_params['size']
hidden_dim = 100 #32
rnn_depth = 2
embedding_dim = 200 #32
prior_low = gen_model_params['theta_range'][0]
prior_high = gen_model_params['theta_range'][1]

# Settings for training
n_train = 50 #5000 #50
n_validation = 10
size_per_theta_train = 100 #2 #100 #20
batch_size = 100
n_epochs = 100
lr = 0.001  # initial learning rate, we have a learning rate scheduler
weight_decay=0.0005
is_train = True
load_dataset = False

# Settings for saving/loading models
model_prefix = 'model/'
model_name = 'formation.model'
checkpoint_range = 10

# Settings for evaluation plots
save_fig = True
show_fig = True
estimator_type = 'posterior_mean'
n_eval_thetas = 40
n_eval_theta_samples = 10

# Settings for debugging
save_network_img = False
img_prefix = 'images/'
