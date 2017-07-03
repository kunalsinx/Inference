from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
import numpy as np
from keras.models import load_model



def train(trainX, trainY):

    Y_train = trainY.reshape(60000,1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))

    Adam = optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=Adam)
    train_Y = np.zeros((60000, 10))
    for i in range(train_Y.shape[0]):
        train_Y[i,Y_train[i]] = 1

    model.fit(trainX, train_Y, batch_size=256, epochs=45)

    model.save('cnn.h5')



def test(testX):

    test_mod = load_model('cnn.h5')
    return np.argmax(test_mod.predict(testX),1)