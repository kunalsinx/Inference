import numpy as np
import train

def main():
	
	hit_shape = np.load("../../dataset/doublets/npy/hit_shape.npy")
	hit_info = np.load("../../dataset/doublets/npy/hit_info.npy")
	target = np.load("../../dataset/doublets/target.npy")
	#train.train(hit_shape, hit_info, target)
	train.load(hit_shape,hit_info)

if __name__ == '__main__':
	main()

