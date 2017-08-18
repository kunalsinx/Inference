#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

int main()
{
    layers::input input_layer1(shape3d(28, 28, 1));
    layers::conv conv1(28, 28, 3, 1, 32);
    layers::max_pool mpool1(26, 26, 32, 2, 2, 2, 2);
    layers::conv conv2(13, 13, 3, 32, 64);
    layers::max_pool mpool2(11, 11, 64, 2, 2, 2, 2);
    layers::input input_layer2(800);
    layers::concat concat({shape3d(1,1,1600),shape3d(1,1,800)});
    layers::fc fc1(2400, 256);
    layers::fc fc2(256, 10);
    activation::relu relu();
    activation::softmax softmax();

    (mpool2, input_layer2) << concat;
    input_layer1 
    << conv1 << relu 
    << mpool1 
    << conv2 <<  
    << mpool2 
    << concat 
    << fc1 << 
    << fc2 << softmax;
    network<graph> net;
    construct_graph(net, { &input_layer1}, { &softmax } );
    std::ofstream ofs("graph_net_example.txt");
    graph_visualizer viz(net, "graph");
    viz.generate(ofs);

}
