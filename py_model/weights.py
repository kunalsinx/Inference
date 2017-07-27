from keras.models import load_model
import numpy as np

def extract(path_model):
	model = load_model(path_model)
	#print len(model.layers)
	print "\nthe architecture........\n"
	j = 1
	print model.summary()
	summary = model.get_config()
	print "\nlayer information........"
	for items in summary:
		#print items['class_name']
		if(items['class_name'] =='Conv2D'):
		 	print "layer",j," = Conv2D"
		 	print "\tnumber of filters = ", items['config']['filters']
		 	print "\tstrides = ", items['config']['strides']
		 	print "\tkernel_size = ", items['config']['kernel_size']
		 	print "\tactivation = ", items['config']['activation'], "\n"
		if(items['class_name'] == 'Dense'):
			print "layer",j," = Dense"
			print "\tunits = ", items['config']['units']
			print "\tactivation = ", items['config']['activation'], "\n"

		if(items['class_name'] == 'Dropout'):
			print "layer",j," = Dropout\n"

		if(items['class_name'] == 'MaxPooling2D'):
			print "layer",j," = MaxPooling2D\n"

		if(items['class_name'] == 'Flatten'):
			print "layer",j," = Flatten\n"
			#print items
		j = j+1

	j = 0


	print "\nExtracting weights and biases...."
	for i in xrange(len(model.layers)):
		weight = model.layers[i].get_weights()
		try:
			j = j+1
			print" weight matrix shape for layer {}: {}".format(j , weight[0].shape)
			print" bias shape for layer {}: {}".format(j , weight[1].shape)
			if j==10:
				np.save("../wght_translation/array.npy", weight[1])
				print weight[1]
		except Exception, e:
			pass

	#model.save_weights("trained-weight.h5")