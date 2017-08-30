import numpy as np
import train

def XOR_dataset():
	table = np.zeros((4,2),dtype = int)

	table[1][1] = 1
	table[2][0] = 1
	table[3][0] = 1
	table[3][1] = 1

	output = np.zeros((4,),dtype = int)
	output[1] = 1
	output[2] = 1
	print table, output

	return table,output

def main():
	x_train, y_train = XOR_dataset()
	np.save("table.npy", x_train)
	np.save("out.npy", y_train)
	#train.train(x_train, y_train)
	train.load(x_train)

if __name__ == '__main__':
	main()

