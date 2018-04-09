# Params and other settings are set here
# Settings are for the generative model as well as the inference engine


# The generative model settings

import networkx as NX

gen_model_params = dict()


gen_model_params['size'] = 20
gen_model_params['network'] = NX.empty_graph(gen_model_params['size'])
gen_model_params['positions'] = NX.random_layout(gen_model_params['network'])

# {
#         'theta_0': 0,  # -0.5,
#         'theta_1': 0.25,  # populationSize,
#         # 'theta_2': 0.525,
#         'theta_3': 1,
#         }

# The inference engine settings

n_hidden = 16  # number of units in each layer of the recurrent unit

NUM_LAYERS = 3  # number of layers in each recurrent unit

OUTPUT_SIZE = 2 # output of the fully connected linear module at the end before the softmax

number_of_features = 1 # the (global) average clustering or transitivity

number_of_classes = 3
