from keras.models import load_model
import numpy as np

def extract(path_model):
	model = load_model(path_model)
	#print len(model.layers)
	print "the architecture\n"
	j = 0
	summary = model.get_config()
	for items in summary:
		#print items['class_name']
		if(items['class_name'] =='Conv2D'):
		 	print "layer = Conv2D"
		 	print "number of filters = ", items['config']['filters']
		 	print "strides = ", items['config']['strides']
		 	print "kernel_size = ", items['config']['kernel_size']
		 	print "activation = ", items['config']['activation'], "\n"
		if(items['class_name'] == 'Dense'):
			print "layer = Dense"
			print "units = ", items['config']['units']
			print "activation = ", items['config']['activation'], "\n"

		if(items['class_name'] == 'Dropout'):
			print "layer = Dropout\n"

		if(items['class_name'] == 'MaxPooling2D'):
			print "layer = MaxPooling2D\n"

		if(items['class_name'] == 'Flatten'):
			print "layer = Flatten\n"
			#print items
			




	print "\nExtracting weights and biases...."
	for i in xrange(len(model.layers)):
		weight = model.layers[i].get_weights()
		try:
			j = j+1
			print" weight matrix shape for layer {}: {}".format(j , weight[0].shape)
			print" bias shape for layer {}: {}".format(j , weight[1].shape)
		except Exception, e:
			j = j-1
			pass

	model.save_weights("weight.h5")