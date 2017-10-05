## Automatic code generation and inference in C++

This is the GSoC project I am doing under CERN-HSF organization with Felice Pantaleo as the mentor.   
The goal of the project is to make a module which reproduces the DNN network originally written in python using keras in C++.     
So the project can be divided in 4 parts     
* Extracting the architecture and important information related to each layers in the network  
* Converting numpy arrays into c++ vectors  
* Code generator for sequential and A+B->C type graph networks which generates the equivalent DNN in C++. 
* Setting weights and biases in each layer after the C++ architecture is produced and hence doing Inference. Setting weights and biases for Dense layer and Convolution layer is supported which can be seen in the file `inference_doublets.cpp`, `inference_xor.cpp` in `cp_model` folder. `inference_doublets.cpp` is more robust and near to completion. Please go throught the Repository Overview below, important for sucessful usage of this module.

### Repository Overview
* `py_model` - contains  
  * `code_gen.py` - extracts the architecture and important information related to each layers in the network and then produces the equivalent network in C++ using the framework of `tiny-dnn`. Weights and biases per layer are stored in the `weights` folder. This script prints - the architecture, layer-wise details (like for a convolution layer it prints the number of filters, strides, kernel size, padding and the activation function used), weight matrix shape for each layer and name of the corresponding layers that is used to define the keras model in C++ non-sequential networks.   
  * `translate.py` - weights stored in the folder `weights` are in `.npy` format. This scripts converts `.npy` file to `.txt` file. The generated files are stored in `txt_weights` folder  
  * `Doublets` - Contains files to build and train the DNN on doublets dataset. On running `main.py`, training will be done and the trained model will be saved in `test_models/saved/models/`. `doublet_translate.py` should be ran for translating the input files in `dataset/doublets/npy` from `.npy` format to `.txt` format which is stored in `dataset/doublets/txt/`.
  * `XOR` - There are two files `main.py` and `train.py`which can be used to train a DNN to learn XOR representation. You can build your DNN model and save it and use it to do the Inference
  * `MNIST` - Currently it contains a CNN model for learning digit classification for MNIST dataset. But one can build their own model. One can edit `main.py` and `train.py` to build their own DNN using keras and train it. Using the code generator, equivalent DNN network can be made in C++.    
* `cp_model` - contains 
  * `target.cpp` - this is the file which is produced after executing `code_gen.py`. The equivalent architecture is defined and saved as binary file named `model` after you compile and run this file.
  * `inference_doublets.cpp` - this file is used to do the inference on doublets dataset which is present in `dataset/doublets/`. Trained weights are loaded into vectors and the model saved after running `target.cpp` are loaded too. Then the weights and biases are set into the Dense layers and Convolution layers. After that predictions are made.
  * `inference_xor.cpp` - this file is used to do the inference for saved trained model for learning XOR representation. Trained weights are loaded into vectors and the model saved after running `target.cpp` are loaded too. Then the weights and biases are set into the Dense layers. After that predictions are made.
  * `inference.cpp` - this file loads the saved trained weights from the folder `txt_weights` and stores them in c++ vec. Then the saved `model` is loaded and the architecture is printed with dimensions for input, output and weights. 
  * `inference_b.cpp` - this file loads the saved trained weights directly from the `.npy` format and stores them in c++ vec.
  * `train_sequential.cpp` - this file conatins an example to make a sequential network using tiny-dnn 
  * `train_graph.cpp` - this file conatins an example to make a graph(A+B->C) network using tiny-dnn
  * `sample_tiny_dnn_train.cpp` - an example to define and make a network using tiny-dnn
* `test_models/saved_models` - this is where the trained model from keras should be saved

### How to use the code
1) git clone this https://github.com/tiny-dnn/tiny-dnn.git and this repo
2) `cd py_model`
3) run `python code_gen.py name_of_the_trained_model_stored_in_test_models/saved_models/`  (this will generate `target.cpp` in `cp_folder` and will create folders `weights` and `txt_weights` if not present and will save the weights as numpy)
4) `python translate.py`  converts `.npy` file to `.txt` file. 
5) cd `cp_model`. Compile `target.cpp` using `g++ target.cpp -Ipath_to_tiny_dnn -o main "-std=c++14" -pthread` and then run `./main`. This will generate `model` binary file and `graph_net_example.txt`(if the network is a graph one, not valid for sequential one). To visualize the graph `dot -Tgif graph_net_example.txt -o graph.gif`. 
6) Depending on the dataset you used. Like if you used doublets dataset then `g++ inference_doublets.cpp -Ipath_to_tiny-dnn/ -o main "-std=c++14" -pthread` and run `./main`.

When the network is sequential, please name the input file as `hit_shape.npy`. When the network has two inputs, please name the input file for branch A as `hit_shape.npy` and for branch B as `hit_info.npy`. 
After this code is used for the first time, `weights` and `txt_weights` folders are made. When re-running the code with a different keras model, please ensure that both these folders are empty( it will work even if you dont delete it but still)  


### Dependencies 
Nothing. All you need is g++ version that supports C++14 and python2

### Demo for automatic code generation and inference for XOR
This is a presentation where I show steps and the outputs of the step https://docs.google.com/presentation/d/1QDXf2t0ysrREG_owA45hQVZtsDVk6iOS8XQIlSMEsoQ/edit?usp=sharing

### Future Work
I am able to set weights and biases into Convolution layer and Dense layer. I have tested this module on doublet dataset and the results were correct. I need to do some finishing work.
 Future works -  
* Making it compatible for python 3
* Improving exception handling
* Loading of input dataset into inference files is not automated yet so this can be done. 
* Adding support for more layers like Batchnorm, deconvolution layer

### Acknowledgement
I would like to thank my mentor Felice Pantaleo for the support and guidance throughtout the project. As well as I would like to thank CERN-HSF and Google for giving me this opportunity to have a valuable,amazing Summer.
