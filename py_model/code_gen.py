from keras.models import load_model
import numpy as np
import os

def extract(path_model):
	model = load_model(path_model)
	#print len(model.layers)
	print "\nthe architecture........\n"
	print model.summary()
	j = 1
	layer_info=[]
	input_shape = []
	layer = []
	summary = model.get_config()
	print "\nlayer information........"
	#print len(summary)
	for items in summary:
		#print items['class_name']
		if(items['class_name'] =='Conv2D'):
		 	print "layer",j," = Conv2D"
		 	print "number of filters = ", items['config']['filters']
		 	print "strides = ", items['config']['strides']
		 	print "kernel_size = ", items['config']['kernel_size']
		 	print "activation = ", items['config']['activation'], "\n"
		 	features = {}
		 	features["number of filters"] = items['config']['filters']
		 	features["strides"] = items['config']['strides']
		 	features["kernel_size"] = items['config']['kernel_size']
		 	features["activation"] = items['config']['activation']
		 	layer_info.append({items['class_name']: features})
		 	#print layer_info
		if(items['class_name'] == 'Dense'):
			print "layer",j," = Dense"
			print "units = ", items['config']['units']
			print "activation = ", items['config']['activation'], "\n"
			features = {}
			features["units"] = items['config']['units']  
			features["activation"] = items['config']['activation']
			layer_info.append({items['class_name']: features})
		
		if(items['class_name'] == 'Dropout'):
			layer.append(items['class_name'])
			print "layer",j," = Dropout\n"

		if(items['class_name'] == 'MaxPooling2D'):
			layer.append(items['class_name'])
			print "layer",j," = MaxPooling2D"
			print "pool size = ", items['config']['pool_size'], "\n"
			features = {}
			features["pool_size"] = items['config']['pool_size'] 
			layer_info.append({items['class_name']: features})

		if(items['class_name'] == 'Flatten'):
			layer.append(items['class_name'])
			print "layer",j," = Flatten\n"
			layer_info.append({items['class_name']: features})
		#print items['class_name']
		input_shape.append((items['class_name'], model.layers[j-1].input_shape))
		#print 
		j = j+1
	#print model.inputs
	#print model.outputs
	print input_shape
	j = 0


	print "\nExtracting weights and biases...."
	for i in xrange(len(model.layers)):
		weight = model.layers[i].get_weights()
		try:
			j = j+1
			print" weight matrix shape for layer {}: {}".format(j , weight[0].shape)
			print" bias shape for layer {}: {}".format(j , weight[1].shape)
			np.save("../wght_translation/weight_"+layer[i]+"_"+str(j%2 + 1)+".npy",weight[0])
			np.save("../wght_translation/bias_"+layer[i]+"_"+str(j%2 + 1)+".npy",weight[1])

		except Exception, e:
			j = j - 1
			pass

	return  layer_info, input_shape

	#model.save_weights("trained-weight.h5")

def code_gen():
	tiny_layers = {"Conv2D":"conv", "Dense":"fc", "MaxPooling2D":"max_pool", }
	tiny_activations = {"relu":"relu", "softmax":"softmax"}

	file = open("goal.cpp","w")

	file.close()



def main():
	layer_info, input_shape = extract('cnn.h5')
	code_gen()

if __name__ == '__main__':
    main()