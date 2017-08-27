from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.models import load_model


def train(x_train, y_train):

	model = Sequential()
	model.add(Dense(2, activation='sigmoid', input_dim = 2))
	model.add(Dense(4, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))

	Adam = optimizers.Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['binary_accuracy'])

	model.fit(x_train, y_train, nb_epoch=500, verbose=2)

	model.save("../../test_models/saved_models/xor.h5")

	print model.predict(x_train)


def load(x_train):

	model = load_model("../../test_models/saved_models/xor.h5")
	print model.predict(x_train)


