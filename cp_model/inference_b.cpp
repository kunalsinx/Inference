#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "tiny_dnn/tiny_dnn.h"
#include "Numpy.hpp"

using namespace tiny_dnn;

int main()
{	
	std::ifstream arch;
	std::vector<std::string> keras_layers;
	try
 	{	
 		arch.open("../weights/layers_with_weights.txt");
 	}

 	catch (std::exception& e)
 	{
 		std::cout << "Error in opening layers.txt";
 		return 0;
 	}
 	char output[100];
 	try 
 	{
 		while (!arch.eof()) 
 		{
 			arch >> output;
 			keras_layers.push_back(output);
 		}
	}
	catch (std::exception& e)
	{
		std::cout << "Error in opening layers.txt";
	}
	arch.close();
	keras_layers.erase(keras_layers.end());

	std::cout <<"Layers with weights...\n" << std::endl;
	for ( auto layer : keras_layers)
	{
		std::cout << layer << std::endl;
	} 
	std::vector<std::vector<int> > shape_w,shape_b;
	std::vector<std::vector<float> >  weight,bias;
	std::vector<float> data;
	std::vector<int> s;
	int i;
	//std::cout << layers[0] << std::endl;
	std::cout << "Loading Numpy arrays into Vectors........";
	std::cout << keras_layers.size() << std::endl;
	for ( i=0; i<keras_layers.size(); i++ )
	{
		std::string file="../weights/weight_";
		file.append(keras_layers[i]);
		file.append(".npy");
		//std::cout << file <<std::endl;
		try
		{
			aoba::LoadArrayFromNumpy(file, s, data);
		}
		catch (std::exception& e)
		{
			return 0;
		}
		weight.push_back(data);
		shape_w.push_back(s);
		//std::cout << file << std::endl;

		file ="../weights/bias_";
		file.append(keras_layers[i]);
		file.append(".npy");
		//std::cout << file <<std::endl;
		try
		{
			aoba::LoadArrayFromNumpy(file, s, data);
		}
		catch (std::exception& e)
		{
			continue;
		}
		bias.push_back(data);
		shape_b.push_back(s);
		// std::cout << data.size() << std::endl;
	
	}

	std::cout << "COMPLETE" << std::endl;
	
	std::cout << "Setting weights and biases...." << std::endl;
	network<graph> cnn;
    cnn.load("../cp_model/model");

    std::cout << "Printing the architecture ....\n";
    
    std::vector<vec_t*> weights;
    for ( i=0; i<cnn.depth();i++)
    {	
		std::cout << "layer type:" << cnn[i]->layer_type() << "\n";
		std::cout << "input:" << cnn[i]->in_data_size() << "(" << cnn[i]->in_data_shape() << ")\n";
		std::cout << "output:" << cnn[i]->out_data_size() << "(" << cnn[i]->out_data_shape() << ")\n";
		std::cout << "layer type:" << cnn[i]->layer_type() << "\n";

		weights = cnn[i]->weights();
		if (weights.size()==0) continue;
    	else 
    	{
			std::cout << "weight dimension:" << weights[0]->size() << std::endl;
			std::cout << "bias dimension:"<< weights[1]->size() << std::endl;
    	}

    }

}
   //  parse_mnist_labels("../dataset/t10k-labels.idx1-ubyte", &test_labels);
  	// parse_mnist_images("../dataset/t10k-images.idx3-ubyte", &test_images, 0, 255, 0, 0);

  	// //std::cout << test_labels[2] << "\n" << std::endl;
  	// for(auto i : test_labels)
  	// {
  	// 	std::cout << i << std::endl;
  	// }

   //  std::vector<vec_t*> weights;
   //  for ( i=0; i<cnn.depth();i++)
   //  {	
   //  	weights = cnn[i]->weights();
   //  	if (weights.size()==0) continue;
   //  	else 
   //  	{
   //  		std::cout << "layer type:" << cnn[i]->layer_type() << "\n";
   //  		std::cout << "input:" << cnn[i]->in_data_size() << "(" << cnn[i]->in_data_shape() << ")\n";
   //  		std::cout << "output:" << cnn[i]->out_data_size() << "(" << cnn[i]->out_data_shape() << ")\n";
   //  		std::cout << "layer type:" << cnn[i]->layer_type() << "\n";
   //  		std::cout << weights[0]->size() << std::endl;
   //  		std::cout << weights[1]->size() << std::endl;
   //  	}

   //  }

    //cnn.test(test_images, test_labels).print_detail(std::cout);
