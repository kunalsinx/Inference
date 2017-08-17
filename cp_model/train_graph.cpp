#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;



int main()
{
	input input_layer_1(shape3d(8, 8, 8	));
	conv conv_2(8, 8, 3, 3, 8, 32, padding::same);
	conv conv_3(8, 8, 3, 3, 32, 32, padding::same);
	max_pool mpool_4(8, 8, 32, 2, 2, 2, 2);
	conv conv_5(4, 4, 3, 3, 32, 64, padding::same);
	conv conv_6(4, 4, 3, 3, 64, 64, padding::same);
	max_pool mpool_7(4, 4, 64, 2, 2, 2, 2);;
	input input_layer_7(36);
	concat concat_8({shape3d(1,1,256),shape3d(1,1,36)});
	fc fc_9(292, 128);
	fc fc_10(128, 64);
	fc fc_11(64, 2);

	(mpool_7, input_layer_7) << concat_8;
	input_layer_1 
	<< conv_2
	<< conv_3  
	<< mpool_4 
	<< conv_5
	<< conv_6
	<< mpool_7 
	<< concat_8 
	<< fc_9 
	<< fc_10
	<< fc_11;
	network<graph> net;
	construct_graph(net, { &input_layer_1}, { &fc_11 } );
	std::ofstream ofs("graph_net_example.txt");
	graph_visualizer viz(net, "graph");
	viz.generate(ofs);

}