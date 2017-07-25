#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void construct_net(network<sequential>& cnn)
{
	using conv    = convolutional_layer;
	using pool    = max_pooling_layer;
	using fc      = fully_connected_layer;
	using relu    = relu_layer;
	using softmax = softmax_layer;

	cnn << conv(28, 28, 3, 1, 32) << relu()  
        << max_pool(26, 26, 32, 2, 2, 2, 2) << relu()                
        << conv(13, 13, 3, 32, 64) << relu() 
        << max_pool(11, 11, 64, 2, 2, 2, 2) << relu()       
        << fc(1600, 256) << relu()                       
        << fc(256, 10) << softmax(); 
	                      
}

void train_cnn(double learning_rate, const int n_train_epochs, const int n_minibatch, core::backend_t backend_type) 

{
  
  network<sequential> cnn;
  construct_net(cnn);

  std::cout << "load models..." << std::endl;

  std::vector<label_t> train_labels, test_labels;
  std::vector<vec_t> train_images, test_images;

  parse_mnist_labels("../dataset/train-labels.idx1-ubyte", &train_labels);
  parse_mnist_images("../dataset/train-images.idx3-ubyte", &train_images, 0, 255, 0, 0);
  parse_mnist_labels("../dataset/t10k-labels.idx1-ubyte", &test_labels);
  parse_mnist_images("../dataset/t10k-images.idx3-ubyte", &test_images, 0, 255, 0, 0);

  for (int i = 0; i < cnn.depth(); i++) 
  {
    std::cout << "#layer:" << i << "\n";
    std::cout << "layer type:" << cnn[i]->layer_type() << "\n";
    std::cout << "input:" << cnn[i]->in_data_size() << "(" << cnn[i]->in_data_shape() << ")\n";
    std::cout << "output:" << cnn[i]->out_data_size() << "(" << cnn[i]->out_data_shape() << ")\n";
  }
  std::cout << "start training" << std::endl;

  progress_display disp(train_images.size());
  timer t;

  adagrad optimizer;
  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "<< t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    tiny_dnn::result res = cnn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  cnn.train<cross_entropy>(optimizer, train_images, train_labels, n_minibatch,
                n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  cnn.test(test_images, test_labels).print_detail(std::cout);
  cnn.save("Naive-CNN-model");
}

static core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {
    "internal", "nnpack", "libdnn", "avx", "opencl",
  };
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<core::backend_t>(i);
    }
  }
  return core::default_engine();
}


int main()
{
  double learning_rate         = 0.1;
  int epochs                   = 3;
  int minibatch_size           = 100;
  core::backend_t backend_type = core::default_engine();

  std::cout << "Running with the following parameters:" << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train_cnn( learning_rate, epochs, minibatch_size, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
	std::cout<<"hello"<<std::endl;
  return 0;
}

