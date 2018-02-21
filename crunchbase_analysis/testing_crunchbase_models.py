# simple test file to test with the outputs of the generative models in with Crunchbase dataset parameters

import settings

from models import *


if __name__ == '__main__':

    params = settings.params
    dynamics = UtilityModel(params)
    # dynamics.plot_time_series()
    sample = dynamics.gen_torch_sample()
    print(sample)

    training_sample = dynamics.gen_torch_data_set(10)
    print(training_sample)
