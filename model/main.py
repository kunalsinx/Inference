import numpy as np
import os
import train_cnn

def load_mnist():
    data_dir = '../dataset'

    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY


def print_digit(digit_pixels, label='?'):
    for i in range(28):
        for j in range(28):
            if digit_pixels[i, j] > 128:
                print '#',
            else:
                print '.',
        print ''

    print 'Label: ', label


def main():
    trainX, trainY, testX, testY = load_mnist()
    print "Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape

    print "\nDigit sample"
    print_digit(trainX[1], trainY[1])

    train_cnn.train(trainX, trainY)
    labels = train_cnn.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print "\nCNN Test accuracy: %lf%%" % accuracy


if __name__ == '__main__':
    main()
