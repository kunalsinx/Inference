from keras.models import load_model
import numpy as np
import os

def extract(path_model):
	try:
		model = load_model(path_model)
	except Exception, e:
		print "Error loading weight file"

	print "\nthe architecture........\n"
	print model.summary()
	j = 1
	layer_info=[]
	input_shape = []
	layer = []
	count_inputs = 0
	summary = model.get_config()
	print "\nlayer information........"

	for items in summary['layers']:
		#print items
		#print items['class_name'] 
		if(items['class_name'] == 'InputLayer'):
			layer.append(items['class_name'])
			print "layer",j," = InputLayer\n"
			features = {}
			layer_info.append({items['class_name']: features})
			count_inputs +=1 

		if(items['class_name'] =='Conv2D'):
			layer.append(items['class_name'])
		 	print "layer",j," = Conv2D"
		 	print "number of filters = ", items['config']['filters']
		 	print "strides = ", items['config']['strides']
		 	print "kernel_size = ", items['config']['kernel_size']
		 	print "Padding = ", str((items['config']['padding']))
		 	print "activation = ", items['config']['activation'], "\n"
		 	features = {}
		 	features["number of filters"] = items['config']['filters']
		 	features["strides"] = items['config']['strides']
		 	features["kernel_size"] = items['config']['kernel_size']
		 	features["activation"] = items['config']['activation']
		 	features["padding"] = str((items['config']['padding']))
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
			print "strides = ", items['config']['strides']
			features = {}
			features["pool_size"] = items['config']['pool_size']
			features["strides"] = items['config']['strides'] 
			layer_info.append({items['class_name']: features})

		if(items['class_name'] == 'Flatten'):
			layer.append(items['class_name'])
			print "layer",j," = Flatten\n"
			features = {}
			layer_info.append({items['class_name']: features})

		if(items['class_name'] == 'Concatenate'):
			layer.append(items['class_name'])
			print "layer",j," = Concatenate\n"
			features = {}
			layer_info.append({items['class_name']: features})

		if(items['class_name'] == 'BatchNormalization'):
			layer.append(items['class_name'])
			print "layer",j," = BatchNormalization\n"
			print "epsilon = ", items['config']['epsilon']
			print "momentum = ", items['config']['momentum']
			features = {}
			features['epsilon'] = items['config']['epsilon']
			features['momentum'] = items['config']['momentum']
			layer_info.append({items['class_name']: features})

		input_shape.append((items['class_name'], model.layers[j-1].input_shape))
		j = j+1
	
	#print layer_info, input_shape
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
	return  layer_info, input_shape, count_inputs


def code_gen(layer_info, input_shape, count_inputs):
	
	if(count_inputs==0):
		tiny_layers = {"Conv2D":"conv", "Dense":"fc", "MaxPooling2D":"max_pool" }
		tiny_activations = {"relu":"relu", "softmax":"softmax"}
		file = open("../cp_model/target.cpp","w")
		file.writelines("#include <iostream>\n")
		file.writelines("#include \"tiny_dnn/tiny_dnn.h\"\n")
		file.writelines("\nusing namespace tiny_dnn;\n")
		file.writelines("using namespace tiny_dnn::activation;\n")
		file.writelines("using namespace tiny_dnn::layers;\n")
		file.writelines("\nint main()\n{\n")
		file.writelines("\tnetwork<sequential> net;\n")
		file.writelines("\tnet ")
		#print len(layer_info), len(input_shape)

		for i in xrange(len(input_shape)):
			name, dim = input_shape[i]
			# layer = layer_info[i]
			# print layer,"\t",name,"\n"
			
			if(name == "Conv2D" ):
				_, h, w, in_d = dim

				layer = layer_info[i]
				# print layer
				s_w, s_h = layer['Conv2D']['strides']
				activation = layer['Conv2D']['activation']
				k_w, k_h = layer['Conv2D']['kernel_size']
				out_d = layer['Conv2D']['number of filters']
				padding = layer['Conv2D']['padding']

				file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"("+str(w)+", "+str(h)+", "+str(k_w)+", "+str(k_h)+", "+str(in_d)+", "+str(out_d)+", padding::"+padding+");")
				file.writelines(" << " + tiny_activations[activation] + "()")

			elif(name == "MaxPooling2D"):
				_, h, w, in_d = dim
				layer = layer_info[i]
				s_w, s_h = layer['MaxPooling2D']['strides']
				p_h, p_w = layer['MaxPooling2D']['pool_size']

				file.writelines("\n\t\t<< "+ tiny_layers[name] +"("+str(w)+", "+str(h)+", "+str(in_d)+", "+str(p_w)+", "+str(p_h)+", "+str(s_w)+", "+str(s_h)+")")

			elif(name == "Dropout"):
				continue

			elif(name == "Flatten"):
				continue

			elif(name == "Dense"):
				#print len(dim)
				layer = layer_info[i]
				activation = layer['Dense']['activation']

				if (len(dim) == 2):
					_, vec = dim
					#print layer
					file.writelines("\n\t\t<< "+ tiny_layers[name] +"("+str(vec)+", "+str(layer["Dense"]["units"])+")")
					file.writelines(" << " + tiny_activations[activation] + "()")
				else :
					print "Only vector input allowed to Dense layers..."
					break

			else:
				#print name
				print "Error: architecture contains layer not supported by code generation yet..."
				continue;
			

		file.writelines(";\n")
		file.writelines("\tnet.save(\"model\");\n")
		file.writelines("}")	
		file.close()

	elif(count_inputs == 2):
		
		tiny_layers = {"InputLayer": "input","Conv2D":"conv", "Dense":"fc", "MaxPooling2D":"max_pool", "Concatenate":"concat", "BatchNormalization":"batch_normalization_layer" }
		tiny_activations = {"relu":"relu", "softmax":"softmax"}
		network = []
		concat_index = 0
		
		file = open("../cp_model/target.cpp","w")
		file.writelines("#include <iostream>\n")
		file.writelines("#include \"tiny_dnn/tiny_dnn.h\"\n")
		file.writelines("\nusing namespace tiny_dnn;\n")
		file.writelines("using namespace tiny_dnn::activation;\n")
		file.writelines("using namespace tiny_dnn::layers;\n")
		file.writelines("\nint main()\n{")

		#print len(layer_info), len(input_shape)
		j = 0
		for i in xrange(len(input_shape)):
			name, dim = input_shape[i]
			j += 1
			# layer = layer_info[i]
			# print layer,"\t",name,"\n"
			
			if(name == "Conv2D" ):
				_, h, w, in_d = dim

				layer = layer_info[i]
				# print layer
				s_w, s_h = layer['Conv2D']['strides']
				activation = layer['Conv2D']['activation']
				k_w, k_h = layer['Conv2D']['kernel_size']
				out_d = layer['Conv2D']['number of filters']
				padding = layer['Conv2D']['padding']

				file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"("+str(w)+", "+str(h)+", "+str(k_w)+", "+str(k_h)+", "+str(in_d)+", "+str(out_d)+", padding::"+padding+");")
				network.append(str(tiny_layers[name])+"_"+str(j)) 

			elif(name == "MaxPooling2D"):
				_, h, w, in_d = dim
				layer = layer_info[i]
				s_w, s_h = layer['MaxPooling2D']['strides']
				p_h, p_w = layer['MaxPooling2D']['pool_size']

				file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"("+str(w)+", "+str(h)+", "+str(in_d)+", "+str(p_w)+", "+str(p_h)+", "+str(s_w)+", "+str(s_h)+");")
				network.append(str(tiny_layers[name])+"_"+str(j))

			elif(name == "Dropout"):
				j -=1
				continue

			elif(name == "Flatten"):
				j -= 1 
				continue

			elif(name == "Concatenate"):
				_, a_in = dim[0]
				_, b_in = dim[1] 
				file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"({shape3d( 1, 1, "+str(a_in)+"), shape3d( 1, 1, "+str(b_in)+")});")
				network.append(str(tiny_layers[name])+"_"+str(j))
				concat_index = j

			elif(name == "InputLayer"):

				if(len(dim)==2):
					_, vec = dim
					j -= 1
					file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"("+str(vec)+");")
					network.append(str(tiny_layers[name])+"_"+str(j))

				elif(len(dim)==4):
					_, h, w, in_d = dim
					file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"(shape3d("+str(w)+", "+str(h)+", "+str(in_d)+"));")
					network.append(str(tiny_layers[name])+"_"+str(j))

			elif(name == "Dense"):
				#print len(dim)
				layer = layer_info[i]
				activation = layer['Dense']['activation']

				if (len(dim) == 2):
					_, vec = dim
					#print layer
					file.writelines("\n\t"+tiny_layers[name]+" "+tiny_layers[name]+"_"+str(j)+"("+str(vec)+", "+str(layer["Dense"]["units"])+");")
					network.append(str(tiny_layers[name])+"_"+str(j))
				else :
					print "Only vectorized input allowed to Dense layers..."
					break

			else:
				#print name
				print "Error: architecture contains layer not supported by code generation yet..."
				j -= 1
				continue;
			
		print "Printing the name of the layers ...\n", network

		file.writelines("\n\n\t("+network[concat_index-2]+", "+network[concat_index-1]+") << "+network[concat_index]+";\n\t")
		file.writelines(network[0])
		for i in xrange(1,len(network)):
			if ( i==concat_index-1 ):
				continue
			file.writelines("\n\t<< "+network[i])
		file.writelines(";")
		file.writelines("\n\n\tnetwork<graph> net;")
		file.writelines("\n\tconstruct_graph(net, { &"+network[0]+", &"+network[concat_index-1]+" }, { &"+network[len(network)-1]+" } );")
		file.writelines("\n\tnet.save(\"model\");")
		file.writelines("\n\tstd::ofstream ofs(\"graph_net_example.txt\");")
		file.writelines("\n\tgraph_visualizer viz(net, \"graph\");")
		file.writelines("\n\tviz.generate(ofs);\n")
		file.writelines("}")	
		file.close()
	
	else:
		print "Network not supported yet ...."



def main():
	
	layer_info, input_shape, count_inputs = extract('../test_models/saved_models/net_and_weights.h5')
	code_gen(layer_info,input_shape, count_inputs)
	# extract('../test_models/keras_model/net_and_weights.h5')


if __name__ == '__main__':
    main()