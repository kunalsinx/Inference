#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

int main()
{
	auto input_1 = std::make_shared<input>(shape3d(8, 8, 8));
	auto conv_2 = std::make_shared<conv>(8, 8, 3, 3, 8, 32, padding::same);
	auto relu_3 = std::make_shared<relu>();
	auto conv_4 = std::make_shared<conv>(8, 8, 3, 3, 32, 32, padding::same);
	auto relu_5 = std::make_shared<relu>();
	auto max_pool_6 = std::make_shared<max_pool>(8, 8, 32, 2, 2, 2, 2);
	auto conv_7 = std::make_shared<conv>(4, 4, 3, 3, 32, 64, padding::same);
	auto relu_8 = std::make_shared<relu>();
	auto conv_9 = std::make_shared<conv>(4, 4, 3, 3, 64, 64, padding::same);
	auto relu_10 = std::make_shared<relu>();
	auto max_pool_11 = std::make_shared<max_pool>(4, 4, 64, 2, 2, 2, 2);
	auto input_12 = std::make_shared<input>(36);
	auto concat_13 = std::make_shared<concat>(std::initializer_list<shape3d>{{ 1, 1, 256}, { 1, 1, 36}});
	auto fc_14 = std::make_shared<fc>(292, 128);
	auto relu_15 = std::make_shared<relu>();
	auto fc_16 = std::make_shared<fc>(128, 64);
	auto relu_17 = std::make_shared<relu>();
	auto fc_18 = std::make_shared<fc>(64, 2);
	auto softmax_19 = std::make_shared<softmax>();

	(max_pool_11, input_12) << concat_13;	//Connecting the branch A..
	input_1
	<< conv_2
	<< relu_3
	<< conv_4
	<< relu_5
	<< max_pool_6
	<< conv_7
	<< relu_8
	<< conv_9
	<< relu_10
	<< max_pool_11;
	//Connecting the branch B..
	input_12;
	//Concatenating both the branches.....
	concat_13
	<< fc_14
	<< relu_15
	<< fc_16
	<< relu_17
	<< fc_18
	<< softmax_19;

	network<graph> net;
	construct_graph(net, { input_1, input_12 }, { softmax_19 } );
	net.save("model");
	std::ofstream ofs("graph_net_example.txt");
	graph_visualizer viz(net, "graph");
	viz.generate(ofs);
}