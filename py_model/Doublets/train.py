import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import load_model, Model


def train(hit_shape, hit_info, target):
	
	input_a = Input(shape=(8, 8, 8))
	input_b = Input(shape=(36,))
	x = Conv2D(32, (3, 3), padding = 'same', activation='relu')(input_a)
	x = Conv2D(32, (3, 3), padding = 'same', activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x)
	x = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	output_a = Flatten()(x)
	
	y = keras.layers.concatenate([output_a, input_b])
	# merge = keras.layers.concatenate([output_a, input_b])
	y = Dense(256, activation = 'relu')(y)
	y = Dense(160, activation = 'relu')(y)
	y = Dense(128, activation = 'relu')(y)
	y = Dense(80, activation = 'relu')(y)
	y = Dense(64, activation = 'relu')(y)
	output_b = Dense(2, activation = 'relu')(y)

	model = Model(inputs = [input_a, input_b], outputs = output_b)
	
	Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	
	model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['categorical_accuracy'])
	model.fit([hit_shape, hit_info], target, epochs=100, verbose=2)

	model.save("../../test_models/saved_models/doublet.h5")

	print model.predict([hit_shape, hit_info])


def load(hit_shape,hit_info):

	model = load_model("../../test_models/saved_models/doublet.h5")
	print model.predict([hit_shape, hit_info])


