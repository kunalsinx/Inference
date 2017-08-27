from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
import numpy as np
from keras.models import load_model
import json
from keras.models import model_from_json



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

    model.fit(trainX, train_Y, batch_size=256, epochs=1)

    #model.save('cnn.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('model.h5')
    model.save('network.h5')



def test(testX):
    print "\nloading pre trained model..."
    test_mod = load_model('cnn.h5')
    return np.argmax(test_mod.predict(testX),1)