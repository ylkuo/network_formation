from GenerativeModel import MixtureOfGaussians
from torch.utils.data import Dataset

class GMMDataset(Dataset):

    def __init__(self, xDim=3, yDim=2, N=2500):
        self.len = N
        gmm = MixtureOfGaussians(dict([]), xDim, yDim)
        [xsamp, ysamp] = gmm.sampleXY(N)
        self.xsamp = xsamp
        self.ysamp = ysamp
        self.xdim = xDim
        self.ydim = yDim
        ysamp_mean = ysamp.mean(axis=0)
        self.ytrain = ysamp - ysamp_mean

    def __getitem__(self, index):
        return (self.xsamp[index], self.ytrain[index])

    def __len__(self):
        return self.len

    def get_dim(self):
    	return (self.xdim, self.ydim)
