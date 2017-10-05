#include <iostream>
#include <fstream>
#include <stdlib.h> 
#include "tiny_dnn/tiny_dnn.h"
#include <string>

using namespace tiny_dnn;

/*txt2vec() reads the weight files stored in "txt_weights" folder
 */

void seq(
	const std::vector<std::vector<double> >& weight,
	const std::vector<std::vector<double> >& bias) 
{
	int i,j,k,l,idx;
	i = j = k = l = idx = 0;
	network<sequential> dnn;
	std::vector<vec_t*> weights, all_weights;
	std::vector<tiny_dnn::float_t> v;

	dnn.load("model");
	
	std::cout << "Printing the architecture ....\n";
	
    for ( i=0; i<dnn.depth();i++)
    {	
    	
		std::cout << "\nlayer type:" << dnn[i]->layer_type() << "\n";
		std::cout << "input:" << dnn[i]->in_data_size() << "(" << dnn[i]->in_data_shape() << ")\n";
		std::cout << "output:" << dnn[i]->out_data_size() << "(" << dnn[i]->out_data_shape() << ")\n";

		weights = dnn[i]->weights();
	
	/*dnn[i]->weights() returns 2D vector if that layer has weights*/

		if (weights.size()==0) continue;
    	else 
    	{	
			std::cout << "weight dimension(after vectorizing):" << weights[0]->size() << std::endl;
			std::cout << "bias dimension(after vectorizing)" << weights[1]->size() << std::endl;	
		/* this is convention followed by tiny-dnn
		 * first the weights followed by biases
		 * weights are vectorized row-wise 
		 */
			v.insert(v.end(), &weight[j][0], &weight[j][weight[j].size()]);
		    v.insert(v.end(), &bias[j][0], &bias[j][bias[j].size()]);

		    std::cout << v.size() << std::endl;

		    dnn[i]->load(v,idx);
		    idx = v.size();
			
			std::cout << "weight dimension(trained):" << weight[j].size() << std::endl;
			std::cout << "bias dimension(trained)" << bias[j].size() << std::endl;

			j++;

	    }

    }

 	std::cout << "Setting weights and biases complete...." << std::endl;

	std::vector<vec_t > x_data; // this is for loading the input

	k = dnn[0]-> in_data_size();
	vec_t data(k);
  	//inp2vec("hit_shape.txt", x_data);
    
    l = 0;
	std::string line;
  	std::ifstream input_file("../dataset/doublets/txt/hit_shape.txt");

  	if (input_file.is_open())
  	{
  		
    	while ( getline (input_file,line) )
    	{

      		data[l] = static_cast<tiny_dnn::float_t>(atof(line.c_str()));
      		l++;
      		if(l==k)
      		{
      			l = 0;
      			x_data.push_back(data);
      		}

      	}
   		input_file.close();
  	}

  	else std::cout <<"Unable to open file";


	std::cout <<"Predicted output...." << std::endl;
	std::cout.precision(std::numeric_limits<double>::digits10 + 1);

	for (int c=0; c < x_data.size(); c++ )
	{
		auto y_vector = dnn.predict(x_data[0]); 
		//std::cout << y_vector[0] << "\t" << y_vector[1] << std::endl;
		for(i=0;i<y_vector.size();i++)
		{
			std::cout << y_vector[i] <<"\t";
		}
		std::cout <<"\n";
	}

}


void graph_net(
	const std::vector<std::vector<double> >& weight,
	const std::vector<std::vector<double> >& bias) 
{
	int i,j,k,l,idx;
	i = j = k = l = idx = 0;
	network<graph> dnn;
	std::vector<vec_t*> weights, all_weights;
	std::vector<tiny_dnn::float_t> v;

	dnn.load("model");
	
	std::cout << "Printing the architecture ....\n";
	
    for ( i=0; i<dnn.depth();i++)
    {	
    	
		std::cout << "\nlayer type:" << dnn[i]->layer_type() << "\n";
		std::cout << "input:" << dnn[i]->in_data_size() << "(" << dnn[i]->in_data_shape() << ")\n";
		std::cout << "output:" << dnn[i]->out_data_size() << "(" << dnn[i]->out_data_shape() << ")\n";

		weights = dnn[i]->weights();
	
	/*dnn[i]->weights() returns 2D vector if that layer has weights*/

		if (weights.size()==0) continue;
    	else 
    	{	
			std::cout << "weight dimension(after vectorizing):" << weights[0]->size() << std::endl;
			std::cout << "bias dimension(after vectorizing)" << weights[1]->size() << std::endl;	
		/* this is convention followed by tiny-dnn
		 * first the weights followed by biases
		 * weights are vectorized row-wise 
		 */
			v.insert(v.end(), &weight[j][0], &weight[j][weight[j].size()]);
		    v.insert(v.end(), &bias[j][0], &bias[j][bias[j].size()]);

		    std::cout << v.size() << std::endl;

		    dnn[i]->load(v,idx);
		    idx = v.size();
			
			std::cout << "weight dimension(trained):" << weight[j].size() << std::endl;
			std::cout << "bias dimension(trained)" << bias[j].size() << std::endl;

			j++;

	    }

    }

 	std::cout << "Setting weights and biases complete...." << std::endl;

	std::vector<std::vector<vec_t >> x_data(2); // this is for loading the input

	k = dnn[0]-> in_data_size();
	vec_t data(k);
  	//inp2vec("hit_shape.txt", x_data);
    
    l = 0;
	std::string line;
  	std::ifstream input_file("../dataset/doublets/txt/hit_info.txt");

  	if (input_file.is_open())
  	{
  		
    	while ( getline (input_file,line) )
    	{
      		data[l] = static_cast<tiny_dnn::float_t>(atof(line.c_str()));
      		l++;
      		if(l==k)
      		{
      			l = 0;
      			x_data[1].push_back(data);
      		}

      	}
   		input_file.close();
  	}

  	else std::cout <<"Unable to open file";

  	std::ifstream input_file2("../dataset/doublets/txt/hit_shape.txt");

  	k = dnn[1]-> in_data_size();
  	data.resize(k);
  	if (input_file2.is_open())
  	{
  		
    	while ( getline (input_file2,line) )
    	{

      		data[l] = static_cast<tiny_dnn::float_t>(atof(line.c_str()));
      		l++;
      		if(l==k)
      		{
      			l = 0;
      			x_data[0].push_back(data);
      		}

      	}
   		input_file2.close();
  	}

  	else std::cout <<"Unable to open file";

	
	std::cout.precision(std::numeric_limits<double>::digits10 + 1);
	
	int batchsize = x_data[0].size();
	std::vector<tensor_t> input(batchsize);
	tensor_t temp(2);

	for (i = 0; i< batchsize; i++)
	{
		for(int j=0; j<2; j++)
		{
			temp[j] = x_data[j][i];
		}

		input[i] = temp;
	}

	std::cout <<"Predicted output...." << std::endl;
	for (i=0; i < batchsize; i++ )
	{
		auto y_vector = dnn.predict(input[i]); 
		for(j=0;j<y_vector[0].size();j++)
		{
			std::cout << y_vector[0][j] <<"\t";
		}
		std::cout <<"\n";
	}

}




void txt2vec(const std::string& file, std::vector<double>& data) {
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
	
	int i,j=0,idx=0,k=0,l=0;
	char output[100];
	std::vector<std::vector<double> >  weight,bias;
	std::vector<double> data;
	std::ifstream arch;
	std::vector<std::string> keras_layers;
/* keras_layers is vector of name of the layers containing 
 * weights along with their order in the network.
 * So the 2D vecs weight abd bias contains layer wise weights from top to bottom
 * so its easier to match and set the weights
 */
	
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

	if(data[0]==0)
	{
		seq(weight,bias);
	}

	else graph_net(weight,bias);
	
}