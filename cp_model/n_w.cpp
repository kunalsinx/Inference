#include <iostream>
#include "Numpy.hpp"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

int main()
{
	std::vector<std::vector<int> > shape_w,shape_b;
	std::vector<std::vector<float> >  weight,bias;
	std::vector<float> data;
	std::vector<int> s;
	int i;
	std::cout << "Loading Numpy arrays into Vectors........";
	for( i=0;;i++)
	{
		std::string file="../weights/weight_";
		file.append("Conv2D");
		file.append("_");
		file.append(std::to_string(i+1));
		file.append(".npy");
		//std::cout << file <<std::endl;
		try
		{
			aoba::LoadArrayFromNumpy(file, s, data);
		}
		catch (std::exception& e)
		{
			break;
		}
		weight.push_back(data);
		shape_w.push_back(s);
		//std::cout << data.size() << std::endl;
	}

	for(int i=0;;i++)
	{
		std::string file="../weights/bias_";
		file.append("Dense");
		file.append("_");
		file.append(std::to_string(i+1));
		file.append(".npy");
		//std::cout << file <<std::endl;
		try
		{
			aoba::LoadArrayFromNumpy(file, s, data);
		}
		catch (std::exception& e)
		{
			break;
		}
		bias.push_back(data);
		shape_b.push_back(s);
		// std::cout << data.size() << std::endl;
	}
	std::cout << "COMPLETE" << std::endl;
	
	std::cout << "Setting weights and biases...." << std::endl;
	network<sequential> cnn;
	std::vector<label_t> test_labels;
  	std::vector<vec_t> test_images;
    cnn.load("../cp_model/model");
    parse_mnist_labels("../dataset/t10k-labels.idx1-ubyte", &test_labels);
  	parse_mnist_images("../dataset/t10k-images.idx3-ubyte", &test_images, 0, 255, 0, 0);
  	std::cout << test_labels.size() << "\t" << test_images.size() << std::endl;
    std::vector<vec_t*> weights;
    for ( i=0; i<cnn.depth();i++)
    {
    	weights = cnn[i]->weights();
    	if (weights.size()==0) continue;
    	else 
    	{
    		std::cout << "layer type:" << cnn[i]->layer_type() << "\n";
    		std::cout << weights[0]->size() << std::endl;
    	}


    }

    cnn.test(test_images, test_labels).print_detail(std::cout);
}