# USAGE
# python train_minivggnet.py

# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import config as config
from mlib.preprocessing import ImageToArrayPreprocessor
from mlib.preprocessing import SimplePreprocessor
from mlib.preprocessing import PatchPreprocessor
from mlib.preprocessing import MeanPreprocessor
from mlib.callbacks import TrainingMonitor
from mlib.io import HDF5DatasetGenerator
from mlib.nn.conv import AlexNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
pp = PatchPreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=None,
	preprocessors=[pp, mp, iap], binarize=True, classes=3)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128,
	preprocessors=[sp, mp, iap], binarize=True, classes=3)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(learning_rate=1e-2)
model = MiniVGGNet.build(width=64, height=64, depth=3,
	classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // 128,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // 128,
	epochs=75,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()