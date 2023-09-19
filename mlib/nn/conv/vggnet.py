# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class VGGNet:
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

		# first CONV
		model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# second CONV
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# third CONV
		model.add(Conv2D(256, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(256, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(256, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# fourth CONV
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# fifth CONV
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(4096))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		model.add(Dense(4096))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
