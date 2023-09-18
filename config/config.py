# specify whether is running locally 
LOCAL_ENV = False

# define the paths to the images directory
TRAIN_IMAGES_PATH = "../strawberries/train"
TEST_IMAGES_PATH = "../strawberries/test"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 3
NUM_VAL_IMAGES = 500 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
if LOCAL_ENV:
    TRAIN_HDF5 = "/media/tiago/OS/hdf5/train.hdf5"
    VAL_HDF5 = "/media/tiago/OS/hdf5/val.hdf5"
    TEST_HDF5 = "/media/tiago/OS/hdf5/test.hdf5"
else:
    TRAIN_HDF5 = "./gdrive/MyDrive/mba/octo/train.hdf5"
    VAL_HDF5 = "./gdrive/MyDrive/mba/octo/val.hdf5"
    TEST_HDF5 = "./gdrive/MyDrive/mba/octo/test.hdf5"

# path to the dataset mean
DATASET_MEAN = "./expert-octo-train-main/output/mean.json"

# paths to the output model file and the
# directory used for storing plots,
# classification reports, etc.
if LOCAL_ENV:   
    MODEL_PATH = "./output/conv.model"
    OUTPUT_PATH = "./output"
else:
    MODEL_PATH = "./gdrive/MyDrive/mba/output/conv.model"
    MODEL_PATH_CHK = "./gdrive/MyDrive/mba/output/conv.model.chk"
    OUTPUT_PATH = "./gdrive/MyDrive/mba/output"

