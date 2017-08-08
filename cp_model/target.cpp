#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

int main()
{
	network<sequential> cnn;
	cnn 
		<< conv(28, 28, 3, 3, 1, 32) << relu()
