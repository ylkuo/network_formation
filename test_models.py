# simple test file for the classes defined in models.py

from models import *



if __name__ == '__main__':

    params = {
        'theta_0': -1,
        'theta_1': 0.25,  # populationSize,
        'theta_2': 0.525,
        'theta_3': 1,
    }

    dynamics = utility_model(params)
    dynamics.plot_time_series()