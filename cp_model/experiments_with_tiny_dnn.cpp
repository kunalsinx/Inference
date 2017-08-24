#include <iostream>
#include "Numpy.hpp"
// #include <typeinfo>
// #include "tiny_dnn/tiny_dnn.h"

// using namespace tiny_dnn;
// using namespace tiny_dnn::layers;
// using namespace activation;

int main()
{
    // auto input1 = std::make_shared<input_layer>(shape3d(28, 28, 1));
    // auto conv1 = std::make_shared<convolutional_layer>(28, 28, 3, 1, 32);
    // auto mpool1 = std::make_shared<max_pooling_layer>(26, 26, 32, 2, 2, 2, 2);
    // auto conv2 = std::make_shared<convolutional_layer>(13, 13, 3, 32, 64);
    // auto mpool2 = std::make_shared<max_pooling_layer>(11, 11, 64, 2, 2, 2, 2);
    // auto input2 = std::make_shared<input_layer>(800);
    // auto concat1 = std::make_shared<concat>(std::initializer_list<shape3d>{{1,1,1600}, {1,1,800}});
    // auto fc1 = std::make_shared<fully_connected_layer>(2400, 256);
    // auto fc2 = std::make_shared<fully_connected_layer>(256, 10);
    // auto relu_1 =  std::make_shared<relu>();
    // //activation::softmax softmax();

    // (mpool2, input2) << concat1;
    // input1 
    // << conv1 << relu_1 
    // << mpool1 
    // << conv2 
    // << mpool2 
    // << concat1 
    // << fc1 
    // << fc2;
    // network<graph> net;
    // construct_graph(net, { input1, input2 }, { fc2 } );
    // std::ofstream ofs("graph_net_example.txt");
    // graph_visualizer viz(net, "graph");
    // viz.generate(ofs);
    std::vector<float> data;
    std::vector<int> s;
    aoba::LoadArrayFromNumpy("../weights/weight_Conv2D_3.npy", s, data);

    for(auto &i : data)
    {
        std::cout << i << ", ";
    }


}
