import numpy as np
import os

def npy2txt(npyfile):

	print npyfile
	x = np.load("../../dataset/doublets/npy/"+npyfile)
	print x.shape
	txtfile = open("../../dataset/doublets/txt/"+npyfile[:-4]+".txt", 'w+')
	
	if len(x.shape) == 4:
		h, w, d, f = x.shape
		for i in xrange(0,h):
			for j in xrange(0,w):
				for k in xrange(0,d):
					for l in xrange(0,f):
						txtfile.writelines("{}".format(x[i][j][k][l]))
						if(i==h-1 and j==w-1 and k==d-1 and l==f-1):
							break

						txtfile.writelines("\n")

	elif len(x.shape) == 3:
		h, w, d = x.shape
		for i in xrange(0,h):
			for j in xrange(0,w):
				for k in xrange(0,d):
					txtfile.writelines("{}".format(x[i][j][k]))
					if(i==h-1 and j==w-1 and k==d-1):
						break

					txtfile.writelines("\n")	

	elif len(x.shape) == 2:
		h, w = x.shape
		for i in xrange(0,h):
			for j in xrange(0,w):
				txtfile.writelines("{}".format(x[i][j]))
				if(i==h-1 and j==w-1):
					break
				txtfile.writelines("\n")

	elif len(x.shape) == 1:

		h = x.shape 
		for i in xrange(0,h[0]):
			txtfile.writelines("{}".format(x[i]))
			if(i==h[0]-1):
				break
			txtfile.writelines("\n")



def main():

	if not os.path.exists("../../dataset/doublets/txt/"):
		os.makedirs("../../dataset/doublets/txt/")
	
	npy2txt("hit_shape.npy")
	npy2txt("hit_info.npy")


if __name__ == '__main__':
    main()