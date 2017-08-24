#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;



int main()
{
	auto input1 = std::make_shared<input_layer>(shape3d(8, 8, 8	));
	auto conv2 = std::make_shared<conv>(8, 8, 3, 3, 8, 32, padding::same);
	auto conv3 = std::make_shared<conv>(8, 8, 3, 3, 32, 32, padding::same);
	auto max_pool4 = std::make_shared<max_pool>(8, 8, 32, 2, 2, 2, 2);
	auto conv5 = std::make_shared<conv>(4, 4, 3, 3, 32, 64, padding::same);
	auto conv6 = std::make_shared<conv>(4, 4, 3, 3, 64, 64, padding::same);
	auto max_pool7 = std::make_shared<max_pool>(4, 4, 64, 2, 2, 2, 2);;
	auto input7 = std::make_shared<input_layer>(36);
	auto concat8 = std::make_shared<concat>(std::initializer_list<shape3d>{{1,1,256}, {1,1,36}});
	auto fc9 = std::make_shared<fc>(292, 128);
	auto fc10  = std::make_shared<fc>(128, 64);
	auto fc11 = std::make_shared<fc>(64, 2);
	auto relu1 = std::make_shared<relu>();
	auto relu2 = std::make_shared<relu>();
	auto relu3 = std::make_shared<relu>();
	auto relu4 = std::make_shared<relu>();
	auto relu5 = std::make_shared<relu>();
	auto relu6 = std::make_shared<relu>();
	auto softmax1 = std::make_shared<softmax>();

	(max_pool7, input7) << concat8;
	input1 
	<< conv2 << relu1
	<< conv3 << relu2 
	<< max_pool4 
	<< conv5 << relu3
	<< conv6 << relu4
	<< max_pool7 
	<< concat8 
	<< fc9 << relu5
	<< fc10 << relu6
	<< fc11 << softmax1;
	network<graph> net;
	construct_graph(net, { input1, input7}, { fc11 } );
	std::ofstream ofs("graph_net_example.txt");
	graph_visualizer viz(net, "graph");
	viz.generate(ofs);

}