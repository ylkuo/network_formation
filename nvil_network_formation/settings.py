# Params and other settings are set here
# Settings are for the generative model as well as the inference engine


# The generative model settings

import networkx as NX

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

n_hidden = 4  # number of units in each layer of the recurrent unit

NUM_LAYERS = 2  # number of layers in each recurrent unit

OUTPUT_SIZE = 3 # output of the fully connected linear module at the end before the softmax

number_of_features = 10 # the (global) average clustering or transitivity

number_of_classes = OUTPUT_SIZE


class_values = [0, 1, 2] #list(range(number_of_classes))