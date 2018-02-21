# Params and other settings are set here
# Settings are for the generative model as well as the inference engine

import pickle
import copy
import numpy as np
import networkx as NX
# The generative model settings


def get_positions_and_network(filename_categories='top_categories.pkl', filename_time_series='observed_time_series.pkl'):

    top_categories = pickle.load(open('./data/' + filename_categories, 'rb'))

    network_time_series= pickle.load(open('./data/' + filename_time_series, 'rb'))

    risk_profiles = dict.fromkeys(
        network_time_series[0].nodes())  # dictionary keyed by node names contains risk profiles

    risk_aggregate = dict.fromkeys(
        network_time_series[0].nodes())  # dictionary keyed by node names contains total profiles

    interest_profile = {key: [] for key in network_time_series[0].nodes()} # dictionary keyed by node names contains
    # number of investments in some categories

    extracted_positions = dict.fromkeys(
        network_time_series[0].nodes())  # dictionary keyed by node names contains position of nodes for visualization

    for i in network_time_series[0].nodes():
        # print(network_time_series[0].node[i]['category']['Software'])
        # print(network_time_series[0].node[i]['category'])
        for j in top_categories:
            interest_profile[i].append(float(network_time_series[0].node[i]['category'][j]))

        # ratio of investment in software
        # net = 0
        # for j in network_time_series[0].node[i]['category'].keys():
        #     net += network_time_series[0].node[i]['category'][j]
        # print('net', net)
        # break
        risk_profiles[i] = []
        risk_profiles[i].append(network_time_series[0].node[i]['round_type']['seed'])
        risk_profiles[i].append(network_time_series[0].node[i]['round_type']['angel'])
        risk_profiles[i].append(network_time_series[0].node[i]['round_type']['A'])
        risk_profiles[i].append(network_time_series[0].node[i]['round_type']['B'])
        risk_profiles[i].append(network_time_series[0].node[i]['round_type']['C'])

        # print(risk_profiles[i])

        risk_aggregate[i] = float(sum(np.array(risk_profiles[i]) * np.array([5, 4, 3, 2, 1])))

        # print(risk_aggregate[i])

        # set position to the network
        extracted_positions[i] = np.array([risk_aggregate[i]] + interest_profile[i])

        # print(positions)
        # positions = nx.random_layout(network_time_series[0])

        # print(extracted_positions)
        initial_network = copy.deepcopy(network_time_series[0])

    return extracted_positions, initial_network


positions, network = get_positions_and_network('top_categories.pkl','observed_time_series.pkl')

params = {
    'network': copy.deepcopy(network),
    # 'theta_0': 0,  # -0.5, # this is a Dc bias in the joint surplus when making edge formation decisions
    'theta_1': 0,  #
    # 'theta_2': 0.525,
    # 'theta_3': np.random.normal(0, 1), #0.5,
    'size': NX.number_of_nodes(copy.deepcopy(network)),
    'input_type': 'transitivity',
    'feature_length': 1,
    'positions': positions,
    'sparsity': 1000*np.sqrt(8 / NX.number_of_nodes(copy.deepcopy(network))),
    'lower_limit': 0,
    'upper_limit': 1,
    }

# The inference engine settings

n_hidden = 16  # number of units in each layer of the recurrent unit

NUM_LAYERS = 3  # number of layers in each recurrent unit

OUTPUT_SIZE = 2  # output of the fully connected linear module at the end before the softmax

number_of_features = 1  # the (global) average clustering or transitivity

BATCH_SIZE = 4

WINDOW_LENGTH = 32