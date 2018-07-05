import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
import numpy as np
import pylab
import pycxsimulator
import sqlite3

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from dateutil import relativedelta
from operator import itemgetter
from time import mktime, strftime, strptime

import pickle

def month_year_iter(start_month, start_year, end_month, end_year):
    ym_start = 12*start_year + start_month - 1
    ym_end = 12*end_year + end_month - 1
    for ym in range(ym_start, ym_end):
        y, m = divmod(ym, 12)
        yield y, m+1

def init_viz():
    global positions, time, time_networks, labeldict
    time = 0
    labeldict = {}
    positions = nx.random_layout(time_networks[time])
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
    pylab.axis('image')
    pylab.title('t = ' + str(time))

def step_viz():
    global time
    if time < len(time_networks) - 1:
        time += 1

class CrunchbaseData(object):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def _get_data_from_table(self, table_name, field):
        cursor = self.conn.cursor()
        results = cursor.execute('SELECT ' + field + ' FROM ' + table_name)
        return results.fetchall()

    def _get_top_countries(self, num_countries=5):
        cursor = self.conn.cursor()
        results = cursor.execute('SELECT COUNT(investor_country_code), investor_country_code FROM investments WHERE investor_country_code <> \'\' GROUP BY investor_country_code ORDER BY COUNT(investor_country_code) DESC LIMIT ' + str(num_countries))
        return results.fetchall()

    def get_investment_network(self):
        results = self._get_data_from_table('investments', '*')
        network = nx.DiGraph()
        all_categories = defaultdict(int)
        for c_link, c_name, c_cat_list, c_country, c_state, c_region, c_city, \
            i_link, i_name, i_country, i_state, i_region, i_city, \
            r_link, r_type, r_code, funded_at, raised_usd in results:
           # create node and edge
           if i_link not in network:
               network.add_node(i_link)
               network.node[i_link]['name'] = i_name
               network.node[i_link]['location'] = \
                   {'country': i_country, 'state': i_state, 'region': i_region, 'city': i_city}
           if c_link not in network:
               network.add_node(c_link)
               network.node[c_link]['name'] = c_name
               network.node[c_link]['location'] = \
                   {'country': c_country, 'state': c_state, 'region': c_region, 'city': c_city}
           if 'category' not in network.node[i_link].keys():
               network.node[i_link]['category'] = defaultdict(int)
           if 'round_type' not in network.node[i_link].keys():
               network.node[i_link]['round_type'] = defaultdict(int)
           network.add_edges_from([(i_link, c_link, {'round': r_link, 'funded_at': funded_at})])
           # build risk profile
           if r_code != '':
               network[i_link][c_link]['round_type'] = r_code
               network.node[i_link]['round_type'][r_code] += 1
           else:
               network[i_link][c_link]['round_type'] = r_type
               network.node[i_link]['round_type'][r_type] += 1
           # build interest profile
           if c_cat_list != '':
               categories = c_cat_list.split('|')
               for cat in categories:
                   network.node[i_link]['category'][cat] += 1
                   all_categories[cat] += 1
        return network, all_categories

    def convert_to_coinvest_network(self, in_network):
        out_network = nx.Graph()
        for node in in_network.nodes:
            in_edges = in_network.in_edges(node)
            investors = [edge[0] for edge in in_edges]
            if len(investors) < 2: continue
            # add investor nodes
            for investor in investors:
                if investor in out_network: continue
                out_network.add_node(investor)
                for attr in in_network.node[investor].keys():
                    out_network.node[investor][attr] = \
                        deepcopy(in_network.node[investor][attr])
            # add co-invest edges
            invest_cat = in_network
            for i1 in investors:
                t1 = strptime(in_network[i1][node]['funded_at'], '%Y-%m-%d')
                for i2 in investors:
                    if i1 == i2: continue
                    t2 = strptime(in_network[i2][node]['funded_at'], '%Y-%m-%d')
                    if t1 != t2: continue
                    if not out_network.has_edge(i1, i2):
                        out_network.add_edge(i1, i2)
                        out_network[i1][i2]['count'] = 0
                        out_network[i1][i2]['time'] = dict()
                    out_network[i1][i2]['count'] += 1
                    # add first co-investment time of a company
                    if node not in out_network[i1][i2]['time'].keys():
                        out_network[i1][i2]['time'][node] = strftime('%Y-%m-%d', t1)
        return out_network

    def generate_time_series(self, in_network, min_coinvest=3, dt=1, filter_by_nodes=None):
        if filter_by_nodes is not None:
            network = in_network.subgraph(filter_by_nodes)
        else:
            network = in_network
        # get the max and min timestamp in this network
        min_time, max_time = None, None
        for i1, i2 in network.edges():
            for c, c_t in network[i1][i2]['time'].items():
                t = strptime(c_t, '%Y-%m-%d')
                if min_time is None and max_time is None:
                    min_time = t
                    max_time = t
                elif t < min_time: min_time = t
                elif t > max_time: max_time = t
        # generate time series by month
        time_series = []
        prev_network = network.copy()
        prev_network.remove_edges_from(network.edges())
        for year, month in month_year_iter(min_time[1], min_time[0], max_time[1], max_time[0]):
            time_str = str(year) + '-' + str(month)
            end_time = strptime(time_str + '-01', '%Y-%m-%d')
            start_time = datetime.fromtimestamp(mktime(end_time)) - \
                relativedelta.relativedelta(months=dt)
            start_time = start_time.timetuple()
            edges = []
            for i1, i2 in network.edges():
                if prev_network.has_edge(i1, i2): continue
                coinvest_times = 0
                end_ct = start_time
                for c, c_t in network[i1][i2]['time'].items():
                    c_t = strptime(c_t, '%Y-%m-%d')
                    if c_t >= start_time and c_t < end_time:
                        coinvest_times += 1
                        if c_t > end_ct: end_ct = c_t
                if coinvest_times >= min_coinvest: edges.append((i1, i2, end_ct))
            edges = sorted(edges, key=itemgetter(2))
            new_network = prev_network.copy()
            if len(edges) > 0:
                new_network.add_edges_from((i1, i2) for i1, i2, ct in edges)
            # exclude network that doesn't change or is empty
            diff = len(new_network.edges()) - len(prev_network.edges())
            if len(new_network.edges()) == 0 or diff == 0:
                continue
            # interpolate the changes
            if diff == 1:
                time_series.append(new_network)
            else:
                diff_edges = list(new_network.edges() - prev_network.edges())
                inter_networks = [new_network]
                for i in range(diff-1):
                    inter_network = inter_networks[0].copy()
                    inter_network.remove_edges_from([diff_edges[i]])
                    inter_networks.insert(0, inter_network)
                time_series.extend(inter_networks)
            prev_network = time_series[-1]
        return time_series

    def get_top_coinvestor_by_country(self, in_network, num_investors=10, countries=['USA']):
        top_coinvestor = []
        for country in countries:
            subgraph = in_network.subgraph([n for n, attrdict in in_network.node.items() \
                if attrdict['location']['country'] == country])
            coinvestors = sorted(subgraph.degree(subgraph.nodes),
                                 key=itemgetter(1), reverse=True)[:num_investors]
            top_coinvestor.extend(coinvestors)
        return top_coinvestor

    def sample_investor_by_country(self, in_network, num_investors=10, countries=['USA']):
        all_investors = {}
        investor_list = []
        for country in countries:
            subgraph = in_network.subgraph([n for n, attrdict in in_network.node.items() \
                if attrdict['location']['country'] == country])
            nodes = []
            weights = []
            for node, degree in subgraph.degree(subgraph.nodes):
                nodes.append(node)
                weights.append(degree)
            weights = np.asarray(weights)
            weights = weights / float(np.sum(weights))
            all_investors[country] = list(np.random.choice(nodes, size=num_investors, replace=False, p=weights))
            investor_list.extend(all_investors[country])
        return all_investors, investor_list

    def generate_time_series_sliding_window(self, in_network, origin='1988-01-01', start='2008-01-01',
                                            end='2015-12-31', min_coinvest=3, window_day=30,
                                            filter_by_nodes=None):
        if filter_by_nodes is not None:
            network = in_network.subgraph(filter_by_nodes)
        else:
            network = in_network
        # filter by the max and min time in this network
        min_time = strptime(origin, '%Y-%m-%d')
        max_time = strptime(end, '%Y-%m-%d')
        investments = []
        for i1, i2 in network.edges():
            for c, c_t in network[i1][i2]['time'].items():
                t = strptime(c_t, '%Y-%m-%d')
                if t < min_time: continue
                if t > max_time: continue
                investments.append((i1, i2, t, c, datetime.strptime(c_t, "%Y-%m-%d")))
        # sort by time
        investments_by_time = sorted(investments, key=itemgetter(2))
        # generate time series by window
        prev_network = network.copy()
        prev_network.remove_edges_from(network.edges())
        time_series = [prev_network]
        investment_in_window = []
        start_time = datetime.strptime(start, '%Y-%m-%d')
        end_time = datetime.strptime(end, '%Y-%m-%d')
        for investment in investments_by_time:
            print('new: ', investment)
            while len(investment_in_window) > 0 and \
                abs((investment[4] - investment_in_window[0][4]).days) > window_day:
                #print('delete: ', investment_in_window[0])
                del investment_in_window[0]
            i1, i2, t, c, c_t = investment
            investment_in_window.append(investment)
            coinvest_times = 0
            for coinvest in investment_in_window:
                if coinvest[0] == i1 and coinvest[1] == i2:
                    coinvest_times += 1
            if prev_network.has_edge(i1, i2): continue
            if coinvest_times < min_coinvest: continue
            if c_t >= start_time and c_t <= end_time:
                # add the edges added during start and end one by one
                new_network = prev_network.copy()
                new_network.add_edges_from([(i1, i2)])
                time_series.append(new_network)
                prev_network = time_series[-1]
            else:
                # add the edges from original to start so that network is not empty
                prev_network.add_edges_from([(i1, i2)])
        return time_series


TIME_SERIES_TYPE = 1 # 0: top investors, 1: sample investors by country

if __name__ == '__main__':
    cb_data = CrunchbaseData('data/crunchbase_2015.db')
    # construct network
    invest_network, categories = cb_data.get_investment_network()
    coinvest_network = cb_data.convert_to_coinvest_network(invest_network)
    # get top co-investors
    print(sorted(coinvest_network.edges(data='count'),
                 key=itemgetter(2), reverse=True)[:20])
    # get top investors
    print(sorted(invest_network.out_degree(invest_network.nodes),
                 key=itemgetter(1), reverse=True)[:20])

    global time_networks
    if TIME_SERIES_TYPE == 0:
        # get time series for top coinvestors
        top_coinvestor = cb_data.get_top_coinvestor_by_country(coinvest_network, num_investors=10, countries=['USA'])
        top_coinvestor = [inv[0] for inv in top_coinvestor]
        # Great recession: start='2007-12-01', end='2009-06-30'
        # All: start='1988-01-01', end='2015-12-31'
        time_series_sliding = \
            cb_data.generate_time_series_sliding_window(coinvest_network, \
                origin='1988-01-01', start='2010-01-01', end='2015-12-31',
                min_coinvest=1, window_day=31, filter_by_nodes=top_coinvestor)
        pickle.dump(time_series_sliding, open('./data/' + 'observed_time_series_sliding_31day_recent_2011_2015_min1.pkl', 'wb'))
        time_networks = time_series_sliding
    elif TIME_SERIES_TYPE == 1:
        # get top 10 countries
        top_countries = cb_data._get_top_countries(num_countries=5)
        for count, country in top_countries:
            print(country, count)
        countries = [country for count, country in top_countries]
        # sample investors from each country
        investors_dict, investors = cb_data.sample_investor_by_country(coinvest_network, num_investors=10, countries=countries)
        #investors = cb_data.get_top_coinvestor_by_country(coinvest_network, num_investors=10, countries=countries)
        #investors = [inv[0] for inv in investors]
        # Dot-com bubble: start='1997-01-01', end='2000-12-31'
        ts1 = cb_data.generate_time_series_sliding_window(coinvest_network, \
            origin='1996-01-01', start='1997-01-01', end='2000-12-31',
            min_coinvest=1, window_day=31, filter_by_nodes=investors)
        # post dot-com bubble?
        ts2 = cb_data.generate_time_series_sliding_window(coinvest_network, \
            origin='1996-01-01', start='2002-01-01', end='2005-12-31',
            min_coinvest=1, window_day=31, filter_by_nodes=investors)
        pickle.dump(ts1, open('./data/' + 'observed_time_series_dotcom.pkl', 'wb'))
        pickle.dump(ts2, open('./data/' + 'observed_time_series_post_dotcom.pkl', 'wb'))
        time_networks = ts2

    # get top categories
    top_categories = sorted(categories.items(), key=itemgetter(1), reverse=True)[:100]
    top_categories = [cat for cat, _ in top_categories]
    print(top_categories)
    pickle.dump(top_categories, open('./data/' + 'top_categories.pkl', 'wb'))
    # visualize time series
    pycxsimulator.GUI().start(func=[init_viz, draw, step_viz])

