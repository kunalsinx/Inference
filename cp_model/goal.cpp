#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

int main()
{	
	network<sequential> cnn;	
	// defining the network
	cnn 
        << conv(28, 28, 3, 3, 1, 32) << relu()  
        << max_pool(26, 26, 32, 2, 2, 2, 2)                
        << conv(13, 13, 3, 3, 32, 64) << relu() 
        << max_pool(11, 11, 64, 2, 2, 2, 2)       
        << fc(1600, 256) << relu()                       
        << fc(256, 10) << softmax();
    // end ..So only this part of the code needs to be generated
    cnn.save("model");
}