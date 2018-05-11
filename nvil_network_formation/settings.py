# Params and other settings are set here
# Settings are for the generative model as well as the inference engine


# The generative model settings

import networkx as NX
import random as RD
import numpy as np

RD.seed(0)
np.random.seed(0)

gen_model_params = dict()


gen_model_params['size'] = 10
gen_model_params['network'] = NX.empty_graph(gen_model_params['size'])
gen_model_params['positions'] = NX.random_layout(gen_model_params['network'])
gen_model_params['feature_length'] = gen_model_params['size']


# {
#         'theta_0': 0,  # -0.5,
#         'theta_1': 0.25,  # populationSize,
#         # 'theta_2': 0.525,
#         'theta_3': 1,
#         }

# The inference engine settings

n_hidden = 32  # number of units in each layer of the recurrent unit

NUM_LAYERS = 2  # number of layers in each recurrent unit

OUTPUT_SIZE = 3  # output of the fully connected linear module at the end before the softmax

number_of_features = gen_model_params['size']  # the (global) average clustering or transitivity

number_of_classes = OUTPUT_SIZE

class_values = [2, 4, 6]  # list(range(number_of_classes))

support = class_values[-1] - class_values[0] + 2

use_exact_posterior = False

load_model = True
load_model_path = './data/model_2000/'
save_model_path = './data/model_3000/'

is_train = True

save_fig = True
show_fig = False
