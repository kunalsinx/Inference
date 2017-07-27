# Inference
Instructions for branch ```py2cpp```   
```cd py_model```   
run ```python main.py``` which outputs   
1. learned weights
2. the architectre
3. important information about different layers required to reproduce the architecture(currently it is being printed but can be saved in a file)
4. shapes of weights and biases
5. saving all the weights in hdf5

```cd cp_model```
contains train.cpp with same architecture as in ```py_model/train_cnn.py```   
to train : ```g++ -I/path_to_tiny-dnn/ train.cpp -o main -std=c++14 -pthread```   
Currently working on translation of weights. Code for it will be up soon

