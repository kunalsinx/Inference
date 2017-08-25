## Automatic code generation in C++

This is the GSoC project I am doing under CERN-HSF organization with Felice Pantaleo as the mentor.   
The goal of the project is to make a module which reproduces the DNN network originally written in python using keras in C++.     
So the project can be divided in 4 parts     
- [x] Extracting the architecture and important information related to each layers in the network  
- [x] Converting numpy arrays into c++ vectors  
- [x] Code generator for sequential and A+B->C type graph networks which generates the equivalent DNN in C++. 
- [ ] Setting weights and biases in each layer after the C++ architecture is produced and hence doing Inference (Work in Progress)   

### Repository Overview
* `py_model` - contains  
  * `code_gen.py` - extracts the architecture and important information related to each layers in the network and then produces the equivalent network in C++ using the framework of `tiny-dnn`. Weights and biases per layer are stored in the `weights` folder. This script prints - the architecture, layer-wise details (like for a convolution layer it prints the number of filters, strides, kernel size, padding and the activation function used), weight matrix shape for each layer and name of the corresponding layers that is use define the keras model in C++   
  * `translate.py` - weights stored in the folder `weights` are in `.npy` format. This scripts converts `.npy` file to `.txt` file. The generated files are stored in `txt_weights` folder
  * `train.py` - you can edit this and build your own DNN using keras and train it. Currently it contains a CNN model for learning digit classification for MNIST dataset. But you can build a DNN model. You have to save the model after training in .h5 format in the folder `test_models/saved_models`    
* `cp_model` - contains 
  * `target.cpp` - this is the file which is produced after executing `code_gen.py`. The equivalent architecture is defined and saved as binary file named `model` after you compile and run this file.
  * `inference.cpp` - this file loads the saved trained weights from the folder `txt_weights` and stores them in c++ vec. 
  * `inference_b.cpp` - this file loads the saved trained weights directly in the `.npy` format and stores them in c++ vec.
  * `train_sequential.cpp` - this file conatins an example to make a sequential network using tiny-dnn 
  * `train_graph.cpp` - this file conatins an example to make a graph(A+B->C) network using tiny-dnn
  * `sample_tiny_dnn_train.cpp` - an example to define and make a network using tiny-dnn
* `test_models/saved_models` - this is where the trained model from keras should be saved

### How to use the code
1) git clone this https://github.com/tiny-dnn/tiny-dnn.git
2) `cd py_model`
3) run `python code_gen.py name_of_the_trained_model_stored_in_test_models/saved_models/`  (this will generate `target.cpp` in `cp_folder` and will create a folder `weights` if not present and will save the weights as numpy)
4) `python translate.py` this will create a folder `txt_weights` if not present and converts `.npy` file to `.txt` file. 
5) cd `cp_model`. Compile `target.cpp` using `g++ target.cpp -Ipath_to_tiny_dnn -o main "-std=c++14" -pthread` and then run `./main`. This will generate `model` binary file and `graph_net_example.txt`(if the network is a graph one). To visualize the graph `dot -Tgif graph_net_example.txt -o graph.gif`. 
6) Now `g++ inference.cpp -Itiny-dnn/ -o main "-std=c++14" -pthread` and run `./main` will read the weight files and load it into c++ vector.

After this code is used for the first time, `weights` and `txt_weights` folders are made. When re-running the code with a different keras model, please ensure that both these folders are empty. 


  
