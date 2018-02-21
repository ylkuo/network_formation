## This is only for 2D visualization of the Crunchbase data.
# The 2D plane axes are the risk attitude and (scaled)
# number of investments in some particular category.

# If you use conda please install pytorch using "conda install -c anaconda pytorch" this installs torch v0.3.0
# as of 2/13/2018. If you use "conda install pytorch-cpu torchvision -c pytorch" a different torch version (0.3.1)
# is installed and there would be some error.

import matplotlib
matplotlib.use('TkAgg')


# from inference import *
# from models import *

import pickle

import random as RD

import pylab

import pycxsimulator

import networkx as nx

import numpy as np

RD.seed()

def init_viz():
    global positions, time, time_networks, labeldict
    time = 0
    labeldict = {}
    positions = process_cruchbase_attributes(time_networks)
    for node in time_networks[time].nodes():
        labeldict[node] = node.replace('/organization/', '').replace('/person/', '')
    # set position to the network
    for t in range(len(time_networks)):
        for name, pos in positions.items():
            time_networks[t].node[name]['position'] = pos

def draw():
    global positions, time, time_networks, labeldict
    pylab.cla()
    nx.draw(time_networks[time],
            pos=positions,
            node_color=[0 for i in time_networks[time].nodes()],
            labels=labeldict,
            with_labels=True,
            edge_color='c',
            cmap=pylab.cm.YlOrRd,
            vmin=0,
            vmax=1)
    # pylab.axis('image')
    pylab.axis('on')
    pylab.xlabel('Risk Attitude')
    pylab.ylabel('E-Commerce Investments (10x)')
    pylab.title('t = ' + str(time))

def step_viz():
    global time
    if time < len(time_networks) - 1:
        time += 1


# 'seed','angle','A','B','C'
def process_cruchbase_attributes(network_timeseries):

    risk_profiles = dict.fromkeys(network_timeseries[0].nodes())  # dictionary keyed by node names contains risk profiles
    risk_aggregate = dict.fromkeys(network_timeseries[0].nodes())  # dictionary keyed by node names contains total profiles
    interest_profile = dict.fromkeys(network_timeseries[0].nodes())  # dictionary keyed by node names contains number of investments in some categories
    positions = dict.fromkeys(network_timeseries[0].nodes())  # dictionary keyed by node names contains position of nodes for visualization

    for i in network_timeseries[0].nodes():
        # print(network_timeseries[0].node[i]['category']['Software'])
        print(network_timeseries[0].node[i]['category'])
        interest_profile[i] = network_timeseries[0].node[i]['category']['E-Commerce']

        # ratio of investment in software
        # net = 0
        # for j in network_timeseries[0].node[i]['category'].keys():
        #     net += network_timeseries[0].node[i]['category'][j]
        # print('net', net)
        # break
        risk_profiles[i] = []
        risk_profiles[i].append(network_timeseries[0].node[i]['round_type']['seed'])
        risk_profiles[i].append(network_timeseries[0].node[i]['round_type']['angel'])
        risk_profiles[i].append(network_timeseries[0].node[i]['round_type']['A'])
        risk_profiles[i].append(network_timeseries[0].node[i]['round_type']['B'])
        risk_profiles[i].append(network_timeseries[0].node[i]['round_type']['C'])
        print(risk_profiles[i])
        risk_aggregate[i] = sum(np.array(risk_profiles[i])*np.array([5,4,3,2,1]))
        print(risk_aggregate[i])
        # set position to the network
        positions[i] = np.array([float(risk_aggregate[i]), 50*float(interest_profile[i])])
        # print(positions)
        # positions = nx.random_layout(network_timeseries[0])
        print(positions)
    return positions

if __name__ == '__main__':

    network_timeseries = pickle.load(open('./data/'+ 'observed_time_series.pkl', 'rb'))
    # visualize time series
    global time_networks
    time_networks = network_timeseries
    pycxsimulator.GUI().start(func=[init_viz, draw, step_viz])
