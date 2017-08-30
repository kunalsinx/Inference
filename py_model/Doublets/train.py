from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import load_model


def train(x_train, y_train):

	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding = 'same', activation='sigmoid', input_shape=(8, 8, 8)))
	model.add(Conv2D(64, (3, 3), padding = 'same', activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), padding = 'same', activation='sigmoid'))
	model.add(Conv2D(64, (3, 3), padding = 'same', activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation = 'sigmoid'))
	model.add(Dense(2, activation = 'relu'))
	
	Adam = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['categorical_accuracy'])

	model.fit(x_train, y_train, epochs=100, verbose=2)

	model.save("../../test_models/saved_models/doublet.h5")

	print model.predict(x_train)


def load(x_train):

	model = load_model("../../test_models/saved_models/doublet.h5")
	print model.predict(x_train)


