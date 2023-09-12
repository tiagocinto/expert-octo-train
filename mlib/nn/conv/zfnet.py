# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class ZFNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# LAYER_01
		model.add(Conv2D(96, (7, 7), strides=(2, 2),
			input_shape=inputShape, padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
		
		# LAYER_02
		model.add(Conv2D(256, (5, 5), strides=(2, 2), padding="valid"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

		# LAYER_03
		# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same"))
		model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))

		# LAYER_04
		# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same"))
		model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))

		# LAYER_05
		# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
		model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
		
		# LAYER_06
		model.add(Flatten())
		model.add(Dense(4096))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# LAYER_07
		model.add(Dense(4096))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# BLOCK softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
