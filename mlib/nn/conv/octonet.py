
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

class OctoNet:
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

		# Block #1
		model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="valid", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

		# Block #2
		model.add(Conv2D(256, (5, 5), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

		# Block #3
		model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))

		# Block #4
		model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))

		# Block #5
		model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

		# Block #6
		model.add(Flatten())
		model.add(Dense(4096))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# Block #7
		model.add(Dense(4096))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
