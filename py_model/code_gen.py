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
			layer.append(items['class_name'])
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
			layer.append(items['class_name'])
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
			features = {}
			layer_info.append({items['class_name']: features})

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
	
	j = 0

	print "\nExtracting weights and biases...."
	for i in xrange(len(model.layers)):
		weight = model.layers[i].get_weights()
		try:
			j = j+1
			print" weight matrix shape for layer {}: {}".format(j , weight[0].shape)
			print" bias shape for layer {}: {}".format(j , weight[1].shape)
			np.save("../weights/weight_"+layer[i]+"_"+str(i+1)+".npy",weight[0])
			np.save("../weights/bias_"+layer[i]+"_"+str(i+1)+".npy",weight[1])

		except Exception, e:
			j = j - 1
			pass

	file = open('../weights/layers.txt', 'w+')
	file.writelines(["%s\n" % item  for item in layer])
	return  layer_info, input_shape

	#model.save_weights("trained-weight.h5")

def code_gen(layer_info, input_shape):
	tiny_layers = {"Conv2D":"conv", "Dense":"fc", "MaxPooling2D":"max_pool", }
	tiny_activations = {"relu":"relu", "softmax":"softmax"}

	file = open("../cp_model/target.cpp","w")
	file.writelines("#include <iostream>\n")
	file.writelines("#include \"tiny_dnn/tiny_dnn.h\"\n")
	file.writelines("\nusing namespace tiny_dnn;\n")
	file.writelines("using namespace tiny_dnn::activation;\n")
	file.writelines("using namespace tiny_dnn::layers;\n")
	file.writelines("\nint main()\n{\n")
	file.writelines("\tnetwork<sequential> cnn;\n")
	file.writelines("\tcnn \n")
	print layer_info, input_shape

	for i in xrange(len(input_shape)):
		name, dim = input_shape[i]
		
		if(name == "Conv2D" ):
			_, h, w, in_d = dim

			layer = layer_info[i]
			s_w, s_h = layer['Conv2D']['strides']
			activation = layer['Conv2D']['activation']
			k_w, k_h = layer['Conv2D']['kernel_size']
			out_d = layer['Conv2D']['number of filters']

			file.writelines("\t\t<< conv("+str(w)+", "+str(h)+", "+str(k_w)+", "+str(k_h)+", "+str(in_d)+", "+str(out_d)+")")
			file.writelines(" << " + activation + "()\n")

		if(name == "MaxPooling2D"):





		else:
			print "Error: architecture contains layer not supported by code generation yet..."
			break;
		
		
		

	file.close()



def main():
	layer_info, input_shape = extract('../test_models/saved_models/cnn.h5')
	code_gen(layer_info,input_shape)

if __name__ == '__main__':
    main()