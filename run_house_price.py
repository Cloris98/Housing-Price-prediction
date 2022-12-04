from data_prep_house import *
from training_model_house import *
from model_evaluation_house import *


if __name__ == '__main__':
    """

    """
    print(
        '****************************',
        'Start using training data to train our model',
        '****************************',
    )

    TRAINING_FILE_PATH = 'train.csv'

    # Get the data
    TRAINING_DATA = Preprocessing(TRAINING_FILE_PATH).process()
    # Get the best train model
    LR, Ridge_CV, RF = TrainModel(TRAINING_DATA).linear_model()
    # Evaluate the best model
    ModelEval(LR, Ridge_CV, RF, TRAINING_DATA[0][1], TRAINING_DATA[0][3]).whole_()


