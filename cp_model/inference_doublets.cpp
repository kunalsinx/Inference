#include <iostream>
#include <fstream>
#include <stdlib.h> 
#include "tiny_dnn/tiny_dnn.h"
#include <string>

using namespace tiny_dnn;

/*txt2vec() reads the weight files stored in "txt_weights" folder
 */
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

/*inp2vec() is to load the input in the form tiny-dnn wants.
 *But there is currently some error with this approach.
 *Simpler approach is used as shown in the main()
 */
// void inp2vec(const std::string& file, std::vector<std::vector<tensor_t> >& data)
// {
// 	int i,j,k,l;
// 	std::string line;
//   	std::ifstream myfile(file);

//   	//std::cout.precision(std::numeric_limits<double>::digits10 + 1);
//   	if (myfile.is_open())
//   	{
  		
//     	while ( getline (myfile,line) )
//     	{

//     		//std::cout << atof(line.c_str()) << std::endl;
//       		data[i][j][k][l++] = static_cast<tiny_dnn::float_t>(atof(line.c_str()));
//       		std::cout << i << j << k << l << std::endl;
//       		if(l%8==0)
//       		{
//       			l = 0;
//       			k++;
//       			if(k%8==0)
//       			{
//       				k = 0;
//       				j++;

//       				if(j%8==0)
//       				{
//       					j = 0;
//       					i++;
//       				}
//       			}
//       		}

//       	}
//    		myfile.close();
//   	}

//   	else std::cout << "Unable to open file";


// }



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
	
	/*cnn[i]->weights() returns 2D vector if that layer has weights*/

		if (weights.size()==0) continue;
    	else 
    	{	
			std::cout << "weight dimension(after vectorizing):" << weights[0]->size() << std::endl;
			std::cout << "bias dimension(after vectorizing)" << weights[1]->size() << std::endl;	

		v.insert(v.end(), &weight[j][0], &weight[j][weight[j].size()]);
	    v.insert(v.end(), &bias[j][0], &bias[j][bias[j].size()]);

	    std::cout << v.size() << std::endl;
		/* this is convention followed by tiny-dnn
		 * first the weights followed by biases
		 * weights are vectorized row-wise 
		 */
	    cnn[i]->load(v,idx);
	    idx = v.size();
		
		std::cout << "weight dimension(trained):" << weight[j].size() << std::endl;
		std::cout << "bias dimension(trained)" << bias[j].size() << std::endl;
	    
	    // std::cout << "Printing weights followed by bias" << std::endl;
	    
	 //    all_weights = cnn[i]->weights();
	 //    for (auto &weigh : all_weights) {
	 //      for (auto &w : *weigh) 
	 //      {
	 //      	std::cout << w << std::endl;
	 //      }
	 //    }
	    
	 //    std::cout << "Printing  actual weights" << idx << std::endl;
		
		// for(auto &k : weight[j])
	 //    {
	 //    	std::cout << k << std::endl;
	 //    }

	 //    std::cout << "Printing  actual bias" << idx << std::endl;

	 //    for(auto &k : bias[j])
	 //    {
	 //    	std::cout << k << std::endl;
	 //    }

		j++;

	    }

    }

 	std::cout << "Setting weights and biases complete...." << std::endl;

 // 	std::vector<std::vector<tensor_t> > x_data(40, std::vector<tensor_t>(8, tensor_t(8, vec_t(8)))) ;

 //  	//inp2vec("hit_shape.txt", x_data);


	std::vector<vec_t > x_data(40, vec_t(512)) ; // this is for loading the input

  	//inp2vec("hit_shape.txt", x_data);
    
    k = l = 0;
	std::string line;
  	std::ifstream input_file("../dataset/doublets/hit_shape.txt");

  	//std::cout.precision(std::numeric_limits<double>::digits10 + 1);
  	if (input_file.is_open())
  	{
  		
    	while ( getline (input_file,line) )
    	{

    		//std::cout << atof(line.c_str()) << std::endl;
      		x_data[k][l++] = static_cast<tiny_dnn::float_t>(atof(line.c_str()));
      		//std::cout << i << j << k << l << std::endl;
      		if(l%512==0)
      		{
      			l = 0;
      			k++;
      		}

      	}
   		input_file.close();
  	}

  	else std::cout << "Unable to open file";


	std::cout << "Predicted output...." << std::endl;

	for (int c=0; c < x_data.size(); c++ )
	{
		auto y_vector = cnn.predict(x_data[0]); 
		std::cout << y_vector[0] << "\t" << y_vector[1] << std::endl;
	}
}