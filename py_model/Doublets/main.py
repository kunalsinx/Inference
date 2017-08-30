import numpy as np
import train

def main():
	
	x_train = np.load("../../dataset/doublets/hit_shape.npy")
	y_train = np.load("../../dataset/doublets/target.npy")
	#train.train(x_train, y_train)
	train.load(x_train)

if __name__ == '__main__':
	main()

