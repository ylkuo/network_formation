import networkx as nx
import sqlite3

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from dateutil import relativedelta
from operator import itemgetter
from time import mktime, strftime, strptime

def month_year_iter(start_month, start_year, end_month, end_year):
    ym_start = 12*start_year + start_month - 1
    ym_end = 12*end_year + end_month - 1
    for ym in range(ym_start, ym_end):
        y, m = divmod(ym, 12)
        yield y, m+1

class CrunchbaseData(object):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def _get_data_from_table(self, table_name, field):
        cursor = self.conn.cursor()
        results = cursor.execute('SELECT ' + field + ' FROM ' + table_name)
        return results.fetchall()

    def get_investment_network(self):
        results = self._get_data_from_table('investments', '*')
        network = nx.DiGraph()
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
        return network

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
                    t2 = strptime(in_network[i1][node]['funded_at'], '%Y-%m-%d')
                    if not out_network.has_edge(i1, i2):
                        out_network.add_edge(i1, i2)
                        out_network[i1][i2]['count'] = 0
                        out_network[i1][i2]['time'] = dict()
                    out_network[i1][i2]['count'] += 1
                    # add first co-investment time of a company
                    if node not in out_network[i1][i2]['time'].keys():
                        if t1 > t2: ts = t1
                        else: ts = t2
                        out_network[i1][i2]['time'][node] = strftime('%Y-%m-%d', ts)
        return out_network

    def generate_time_series(self, in_network, filter_by_nodes=None):
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
        time_series = dict()
        for year, month in month_year_iter(min_time[1], min_time[0], max_time[1], max_time[0]):
            time_str = str(year) + '-' + str(month)
            start_time = strptime(time_str + '-01', '%Y-%m-%d')
            end_time = datetime.fromtimestamp(mktime(start_time)) + \
                relativedelta.relativedelta(months=1)
            end_time = end_time.timetuple()
            edges = [(i1, i2) for i1, i2 in network.edges()
                     for c, c_t in network[i1][i2]['time'].items()
                     if strptime(c_t, '%Y-%m-%d') < end_time]
            time_series[time_str] = network.edge_subgraph(edges).copy()
        return time_series


if __name__ == '__main__':
    cb_data = CrunchbaseData('data/crunchbase_2015.db')
    invest_network = cb_data.get_investment_network()
    coinvest_network = cb_data.convert_to_coinvest_network(invest_network)
    # get top co-investors
    print(sorted(coinvest_network.edges(data='count'),
                 key=itemgetter(2), reverse=True)[:20])
    # get top investors
    print(sorted(invest_network.out_degree(invest_network.nodes),
                 key=itemgetter(1), reverse=True)[:20])
    # get time series for top coinvestors
    top_coinvestor = sorted(coinvest_network.degree(coinvest_network.nodes),
                            key=itemgetter(1), reverse=True)[:20]
    top_coinvestor = [inv[0] for inv in top_coinvestor]
    time_series = cb_data.generate_time_series(coinvest_network,
                                               top_coinvestor)

