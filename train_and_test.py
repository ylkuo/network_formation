## This file trains and tests the RNN in inference.py with data sets generated by models.py

from inference import *
from models import *

RD.seed()

if __name__ == '__main__':

    dynamics = utility_model(settings.params)

    model = RNN()

    model.empty_losses()

    print('pretraining performance on the traning set')
    training_sample = dynamics.genTorchDataset(200)
    print(100 * model.evaluateAveragePerformance(training_sample))

    print(training_sample)

    print('doTraining on the traning set')

    model.doTraining(training_sample, batch_size = 2, window_length_loss=8, verbose = True ,
                     save = True , file_name = 'model_tuned.pkl')
    print('perforamce on Trainging set AFTER')
    print(100 * model.evaluateAveragePerformance(training_sample))
    model.plot_losses()
    model.save_losses()
    # print(model.training_losses)


    print('performance on Test set after training')

    test_sample = dynamics.genTorchDataset(100)
    print(100 * model.evaluateAveragePerformance(test_sample))