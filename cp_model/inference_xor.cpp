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


}




int main()
{	
	
	int i,j=0,idx=0;
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
	std::cout << " Size of weight vector : "<< weight[0].size() << std::endl << "Size of bias vector : " << bias[0].size() << std::endl;
	network<sequential> cnn;
    cnn.load("../cp_model/model");
	std::vector<vec_t*> weights, all_weights;
	std::vector<tiny_dnn::float_t> v;
	
	std::cout << "Printing the architecture ....\n";
	
    for ( i=0; i<cnn.depth();i++)
    {	
    	
		std::cout << "\nlayer type:" << cnn[i]->layer_type() << "\n";
		std::cout << "input:" << cnn[i]->in_data_size() << "(" << cnn[i]->in_data_shape() << ")\n";
		std::cout << "output:" << cnn[i]->out_data_size() << "(" << cnn[i]->out_data_shape() << ")\n";

		weights = cnn[i]->weights();
		if (weights.size()==0) continue;
    	else 
    	{	
			std::cout << "weight dimension(after vectorizing):" << weights[0]->size() << std::endl;
			std::cout << "bias dimension(after vectorizing)" << weights[1]->size() << std::endl;	

		v.insert(v.end(), &weight[j][0], &weight[j][weight[j].size()]);
	    v.insert(v.end(), &bias[j][0], &bias[j][bias[j].size()]);

	    //std::cout << v.size() << std::endl;

	    cnn[i]->load(v,idx);
	    idx = v.size();
	    
	    std::cout << "Printing weights followed by bias" << std::endl;
	    
	    all_weights = cnn[i]->weights();
	    for (auto &weigh : all_weights) {
	      for (auto &w : *weigh) 
	      {
	      	std::cout << w << std::endl;
	      }
	    }
	    
	    std::cout << "Printing  actual weights" << idx << std::endl;
		
		for(auto &k : weight[j])
	    {
	    	std::cout << k << std::endl;
	    }

	    std::cout << "Printing  actual bias" << idx << std::endl;

	    for(auto &k : bias[j])
	    {
	    	std::cout << k << std::endl;
	    }

			j++;

	    }

    }

  	std::cout << "Setting weights and biases complete...." << std::endl;

  	std::vector<vec_t> x_data {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}} ;
  
  	std::cout << x_data.size()<< std::endl;

	std::cout << "Predicted output...." << std::endl;  
	
	for (int c=0; c < x_data.size(); c++ )			
	{	auto y_vector = cnn.predict(x_data[c]);


		  	std::cout << y_vector[0] << std::endl;

	}
}