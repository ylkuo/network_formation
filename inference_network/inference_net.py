import settings
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Uniform
from truncated_normal import TruncatedNormal

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

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output[:,-1,:])
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
        self.ts_embedding = TimeSeriesEmbedding(settings.n_features, settings.hidden_dim,
                                                settings.rnn_depth, settings.embedding_dim)
        self.proposal_layer = ProposalUniformTruncatedNormal(settings.embedding_dim)

    def forward(self, input):
        proposal_input, _ = self.ts_embedding(input)
        proposal_output = self.proposal_layer(proposal_input,
                                              torch.tensor(settings.prior_low).type(settings.dtype),
                                              torch.tensor(settings.prior_high).type(settings.dtype))
        return proposal_output
