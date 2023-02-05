import os

ROOT_DIR = os.getcwd() + "/"
DATASET_PATH = ROOT_DIR + 'headsegmentation_dataset_ccncsa/'
DRESSCODE_PATH_ON_LAB_SERVER = ...
WEIGHTS_PATH = 'models/weights/'
PLOTS_PATH = 'models/plots/'
DEMO_PATH = 'models/demo/'
CHECKPOINTS_PATH = 'models/training_best/'
HPO_PATH = 'models/hpo/'
PREPROCESSING_PATH = 'models/preprocessing/'
LOSS_SELECTION_PATH = 'models/loss_selection/'

# Name of index file of dataset (file .xml in folder DATASET_PATH).
DATASET_INDEX_NAME = 'training.xml'

# Mean and standard deviation for transform Normalize, to apply to input images when passing them to a model.
NORMALIZE_MEAN = [0.3954, 0.3269, 0.2831]
NORMALIZE_STD = [0.2513, 0.2356, 0.2309]

# Weights assigned to each class in the dataset, representing their importance.
CLASS_WEIGHTS = [0.3762, 0.9946, 0.9974, 0.9855, 0.7569, 0.9140, 0.9968, 0.9936, 0.9989, 0.9893, 0.9968]
