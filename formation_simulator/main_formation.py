# Simple Network Dynamics simulator in Python
#
# *** Network Formation ***
# Based on: Hiroki Sayama's Python file for Network Epidemics
# sayama@binghamton.edu

import matplotlib

matplotlib.use('TkAgg')

import pylab as PL
import numpy as np
import random as RD
import scipy as SP
import networkx as NX
import pycxsimulator
RD.seed()

populationSize = 50
sparsity = np.sqrt(8/populationSize)
print(sparsity)
theta = [-1,0.25,0.35,1]

def init():
    global time, network, positions, nextNetwork, potential_edge_attributes

    time = 0


    network = NX.empty_graph(populationSize)

    positions = NX.random_layout(network)

    potential_edge_keys = list(network.edges()) + list(NX.non_edges(network))
    potential_edge_attributes = dict.fromkeys(potential_edge_keys)
    print(potential_edge_attributes)

    for i in network.nodes():
        network.node[i]['position'] = positions[i]
        network.node[i]['attribute'] = np.random.binomial(1,0.5)

    for potential_edge in potential_edge_keys:
        # network[edge[0]][edge[1]]['attribute'] = np.random.normal(0, theta[3])
        potential_edge_attributes[potential_edge] = np.random.normal(0, theta[3])
        # network[edge[0]][edge[1]]['attribute'] = np.random.normal(0,theta[3])
        # print(network[edge[0]][edge[1]]['attribute'])

    nextNetwork = network.copy()


def draw():
    PL.cla()
    NX.draw(network,
            pos=positions,
            node_color=[0 for i in network.nodes()],
            with_labels=False,
            edge_color='c',
            cmap=PL.cm.YlOrRd,
            vmin=0,
            vmax=1)
    PL.axis('image')
    PL.title('t = ' + str(time))


def step():
    global time, network, nextNetwork, potential_edge_attributes

    time += 1
    candidate_edges = list(NX.non_edges(network))
    RD.shuffle(candidate_edges)
    for candidate_edge in candidate_edges:
        if successful_edge_formation(candidate_edge):
            nextNetwork.add_edge(*candidate_edge)
            break

    network, nextNetwork = nextNetwork, nextNetwork

def successful_edge_formation(candidate_edge):
    global time, network, nextNetwork, potential_edge_attributes
    print(network.node[candidate_edge[0]]['position'])
    print(network.node[candidate_edge[1]]['position'])
    print(np.linalg.norm(network.node[candidate_edge[0]]['position'] - network.node[candidate_edge[1]]['position']))
    distance = np.linalg.norm(network.node[candidate_edge[0]]['position'] - network.node[candidate_edge[1]]['position'])
    list_common_neighbirs = list(NX.common_neighbors(network, candidate_edge[0], candidate_edge[1]))
    edge_value = theta[0] + theta[1]*(network.node[candidate_edge[0]]['attribute'] + network.node[candidate_edge[1]]['attribute']) + \
        theta[2]*(len(list_common_neighbirs)) + potential_edge_attributes[candidate_edge] - (1/sparsity)*distance
    print(theta[2]*(len(list_common_neighbirs)))
    print(candidate_edge)
    print(edge_value)
    return (edge_value > 0)

pycxsimulator.GUI().start(func=[init, draw, step])
