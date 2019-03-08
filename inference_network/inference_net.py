import math
import settings
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Uniform
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from truncated_normal import TruncatedNormal

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='uniform'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            self.reset_parameters_uniform()
        elif init == 'xavier':
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_depth, output_dim):
        super(TimeSeriesEmbedding, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, rnn_depth, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU(0.1)
        # init GRU parameters: orthogonal initialization of recurrent weights
        self.rnn.reset_parameters()
        for _, hh, _, _ in self.rnn.all_weights:
            for i in range(0, hh.size(0), self.rnn.hidden_size):
                nn.init.orthogonal_(hh[i:i + self.rnn.hidden_size])

    def forward(self, features, in_sequences, seq_lengths, hidden=None):
        packed_input = pack_padded_sequence(in_sequences, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        last_out = []
        for i in range(output.shape[0]):
            last_out.append(output[i,input_sizes[i]-1,:])
        last_out = torch.stack(last_out)
        output = self.fc(last_out)
        output = self.relu(output)
        return output, hidden

class GCTimeSeriesEmbedding(nn.Module):
    def __init__(self, gc_dim, hidden_dim, rnn_depth, output_dim):
        super(GCTimeSeriesEmbedding, self).__init__()
        self.gc = GraphConvolution(2, gc_dim)
        self.fc1 = nn.Linear(gc_dim*settings.n_nodes, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, rnn_depth, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU(0.1)
        # init GRU parameters: orthogonal initialization of recurrent weights
        self.rnn.reset_parameters()
        for _, hh, _, _ in self.rnn.all_weights:
            for i in range(0, hh.size(0), self.rnn.hidden_size):
                nn.init.orthogonal_(hh[i:i + self.rnn.hidden_size])

    def forward(self, input, adjs, seq_lengths, hidden=None):
        rnn_in = []
        for i in range(adjs.shape[0]):  # batch
            seq = []
            for j in range(adjs.shape[1]):  # time step
                gc_out = self.gc(input[i], adjs[i,j,:,:])
                gc_out = gc_out.view(gc_out.shape[0]*gc_out.shape[1])
                gc_out = self.fc1(gc_out)
                seq.append(gc_out)
            rnn_in.append(torch.stack(seq))
        rnn_in = torch.stack(rnn_in)
        packed_input = pack_padded_sequence(rnn_in, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        last_out = []
        for i in range(output.shape[0]):
            last_out.append(output[i,input_sizes[i]-1,:])
        last_out = torch.stack(last_out)
        output = self.fc2(last_out)
        output = self.relu(output)
        return output, hidden

class ProposalNormalNormal(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ProposalNormalNormal, self).__init__()
        self._output_dim = output_dim
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, self._output_dim * 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, prior_mean, prior_stddev):
        batch_size = x.size()[0]
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:, :self._output_dim]
        stddevs = torch.exp(x[:, self._output_dim:])
        prior_means = torch.stack([prior_mean for _ in range(batch_size)]).unsqueeze(1)
        prior_stddevs = torch.stack([prior_stddev for _ in range(batch_size)]).unsqueeze(1)
        means = prior_means + (means * prior_stddevs)
        return Normal(means, stddevs)

class ProposalUniformTruncatedNormal(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ProposalUniformTruncatedNormal, self).__init__()
        self._output_dim = output_dim
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, self._output_dim * 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, prior_low, prior_high):
        batch_size = x.size(0)
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:, :self._output_dim]
        stddevs = torch.exp(x[:, self._output_dim:])
        prior_lows = torch.stack([prior_low for _ in range(batch_size)]).unsqueeze(1)
        prior_highs = torch.stack([prior_high for _ in range(batch_size)]).unsqueeze(1)
        means = prior_lows.view(batch_size, -1).expand_as(means) + (means * (prior_highs - prior_lows).view(batch_size, -1).expand_as(means))
        return TruncatedNormal(means, stddevs, low=prior_lows, high=prior_highs,
                               clamp_mean_between_low_high=True)


class InferenceNetwork(nn.Module):
    def __init__(self):
        super(InferenceNetwork, self).__init__()
        if settings.gen_model_params['input_type'] == 'adjacencies':
            self.ts_embedding = GCTimeSeriesEmbedding(settings.n_features, settings.hidden_dim,
                                                      settings.rnn_depth, settings.embedding_dim)
        elif settings.gen_model_params['input_type'] == 'degree_sequence':
            self.ts_embedding = TimeSeriesEmbedding(settings.n_features, settings.hidden_dim,
                                                    settings.rnn_depth, settings.embedding_dim)
        else:
            raise NotImplementedError
        self.proposal_layer = ProposalUniformTruncatedNormal(settings.embedding_dim)

    def forward(self, features, in_sequences, seq_lengths):
        proposal_input, _ = self.ts_embedding(features, in_sequences, seq_lengths)
        proposal_output = self.proposal_layer(proposal_input,
                                              torch.tensor(settings.prior_low).type(settings.dtype),
                                              torch.tensor(settings.prior_high).type(settings.dtype))
        return proposal_output
