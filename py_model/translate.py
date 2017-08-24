import numpy as np
import os

def npy2txt(npyfile):

	print npyfile
	x = np.load("../weights/"+npyfile)
	txtfile = open("../txt_weights/"+npyfile[:-4]+".txt", 'w+')
	
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
	
	print "Converting weights to txt format .....\n"
	
	for file in os.listdir("../weights"):
	    if file.endswith(".npy"):
	        npy2txt(file)

	print "Translation complete..."

if __name__ == '__main__':
    main()