from keras.models import load_model
import numpy as np

def extract(path_model):
	model = load_model(path_model)
	#print len(model.layers)
	print "\nExtracting weights and biases...."
	j = 0
	for i in xrange(len(model.layers)):
		weight = model.layers[i].get_weights()
		try:
			j = j+1
			print" weight matrix shape for layer {}: {}".format(j , weight[0].shape)
			print" bias shape for layer {}: {}".format(j , weight[1].shape)
		except Exception, e:
			j = j-1
			pass