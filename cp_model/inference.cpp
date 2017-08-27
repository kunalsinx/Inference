#include <iostream>
#include <fstream>
#include <stdlib.h> 
#include "tiny_dnn/tiny_dnn.h"
#include <string>
using namespace tiny_dnn;

void txt2vec(const std::string& file, std::vector<double>& data)
{
	std::string line;
  	std::ifstream myfile(file);

  	//std::cout.precision(std::numeric_limits<double>::digits10 + 1);
  	if (myfile.is_open())
  	{
  		
    	while ( getline (myfile,line) )
    	{

    		//std::cout << atof(line.c_str()) << std::endl;
      		data.push_back(atof(line.c_str()));
    	}
   		myfile.close();
  	}

  	else std::cout << "Unable to open file";

  	//   	for(auto i : data)
  	// {
  	// 	std::cout << i << std::endl;
  	// }


}




int main()
{	
	int i;
	char output[100];
	std::vector<std::vector<double> >  weight,bias;
	std::vector<double> data;
	std::ifstream arch;
	std::vector<std::string> keras_layers;
	
	try
 	{	
 		arch.open("../weights/layers_with_weights.txt");
 	}

 	catch (std::exception& e)
 	{
 		std::cout << "Error in opening layers_with_weights.txt";
 		return 0;
 	}

	while (!arch.eof()) 
	{
		arch >> output;
		keras_layers.push_back(output);
	}

	arch.close();
	keras_layers.erase(keras_layers.end());
	
	//std::cout << layers[0] << std::endl;
	std::cout << "Coverting Numpy arrays into Vectors........";
	//std::cout << keras_layers.size() << std::endl;
	for ( i=0; i<keras_layers.size(); i++ )
	{
		std::string file="../txt_weights/weight_";
		file.append(keras_layers[i]);
		file.append(".txt");
		//std::cout << file <<std::endl;
		try
		{
			txt2vec(file, data);
		}
		catch (std::exception& e)
		{
			std::cout << "unknown txt file detected...." << std::endl;
			continue;
		}
		
		weight.push_back(data);
		std::cout << file << std::endl;
		std::cout << "data size : "<< data.size() << std::endl;	
		data.clear();

		file ="../txt_weights/bias_";
		file.append(keras_layers[i]);
		file.append(".txt");
		std::cout << file <<std::endl;
		try
		{
			txt2vec(file, data);
		}
		catch (std::exception& e)
		{
			continue;
		}
		bias.push_back(data);
		std::cout << "data size : "<< data.size() << std::endl;
		data.clear();
	}

	
	std::cout << "COMPLETE" << std::endl;
	
	//std::cout << "Setting weights and biases...." << std::endl;
	std::cout << " Size of weight vector : "<< weight.size() << std::endl << "Size of bias vector : " << bias.size() << std::endl;

	std::string file="../weights/input_count.txt";

	try
	{
		txt2vec(file, data);
	}
	catch (std::exception& e)
	{
		std::cout << "Error in opening input_count.txt" << std::endl;
		return 1;
	}

	//std::cout << data[0] << std::endl;

	if(data[0]>=1)
	{
		network<graph> cnn;

		cnn.load("../cp_model/model");
		std::vector<vec_t*> weights;

		std::cout << "Printing the architecture ....\n";

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
				std::cout << "weight dimension(after vectorizing):" << weights[0]->size() << std::endl;
				std::cout << "bias dimension(after vectorizing)" << weights[1]->size() << std::endl;
	    	}

	    }
	}
	
	else
	{
		network<sequential> cnn;

		cnn.load("../cp_model/model");
		std::vector<vec_t*> weights;

		std::cout << "Printing the architecture ....\n";

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
				std::cout << "weight dimension(after vectorizing):" << weights[0]->size() << std::endl;
				std::cout << "bias dimension(after vectorizing)" << weights[1]->size() << std::endl;
	    	}

	    }

	}
	

}