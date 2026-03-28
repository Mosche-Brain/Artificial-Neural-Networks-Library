# YANN - Yet Another Neural Networks library
### Ligtweight, minimal and easy to use C++ library for machine learning

## ⬇️ Installaction and building process

### 🐧 Linux and GNU/Linux

#### requirements
* eigen3 devel
* OpenMP
* G++ compiler
* CMake
* CPU with at least one core
```bash
git clone [this repo]
cd [this repo folder]
mkdir build
cd build
cmake ..
make
```

### 🪟 Windows and MacOS
Probably almost, just like in linux and GNU/Linux (I guess).

## 💡 Examples of usage

### Creating and fitting sequential model
```cpp
#include <YANN/Models/Sequential.hpp>

using namespace YANN::Models;

int main()
{
    int input_layer_size = 2;
    int hidden_layer_size = 69;
    int outut_layer_size = 2;
    Sequential model({Layers::Input::createUnique(input_layer_size),
			          Layers::Dense::createUnique(hidden_layer_size, "relu")
			          Layers::Dense::createUnique(outut_layer_size, "sigmoid")}); 
                      /* you can also put tanh, gelu, softmax, etc */

    int number_of_samples = 3;
	YANN::matrix_t x_train(number_of_samples, input_layer_size);
	YANN::matrix_t y_train(number_of_samples, input_layer_size);

    /* Filling training data */
    YANN::numeric_t learning_rate = 0.1;
    int epochs = 200;

	model->fit(x_train, y_train, learning_rate, epochs);
}
```

### Creating and fitting perceptron
```cpp
#include <YANN/Models/Perceptron.hpp>

using namespace YANN::Models;

int main()
{
    int perceptron_input_size = 2;
    Perceptron model(perceptron_input_size, "relu");

    int number_of_samples = 800;
	YANN::matrix_t x_train(number_of_samples, perceptron_input_size);
	YANN::vector_t y_train(number_of_samples);

    /* Filling training data */

    model.fit(x_train, y_train,, learning_rate, epochs);
}
```

### Defining own layer types
```cpp
#include "MyLayer.hpp"

MyLayer::MyLayer(int layerSize)
{
    /* Initializing layer params */

    this->_initialized_ = true;
}

YANN::matrix_t MyLayer::forward(const YANN::matrix_t& input)
{
    this->outputs = /* ... */;
    return outputs;   
}

YANN::matrix_t MyLayer::backward(const YANN::matrix_t deltaOutput)
{
    /* your implementation of backward pass */
}

void MyLayer::update_weights(YANN::numeric_t rate)
{
    /* Your method to updating weights */
}

std::unique_ptr<LayerBase> MyLayer::createUnique(int layerSize)
{
    /* Layer class fabric method */
    return std::make_unique<MyLayer>(layerSize);
}
```
You can find full API documentation [there](www.amogus.org)


## 🔨 Features
* ✅ Sequential models
* ✅ Dense layers
* ✅ Working backward pass
* ✅ Adjustable numbers precision
* ✅ Parallel CPU Computatiopns
* ⚠️ [Dedicated graphical envionment](https://github.com/Czuowuek-SOS/MLStudio) (work in progress)
* ❌ OneAPI support
* ❌ CUDA support
* ❌ ROCm support
* ❌ Recurent Neural Networks
* ❌ Transformers
* ❌ Build in telemetry
* ❌ Outperform TensorFlow

## 🧷 Credits and used technologies
* [Eigen3](https://eigen.tuxfamily.org/) - C++ Linear Algebra library
* [OpenMP](https://www.openmp.org/) - Multiprocessing interface for C/C++
* [Sarvel](https://sarvel.xyz/) - Literally Digital God and my friend who designs 16 bit CPU's
* [Jakub Gładysz](https://sarvel.xyz/) - PhD at PWr


## 🥱 Other useless informations
### 🧼 Generative AI assistance in this project:
* Name refactoring tasks
* Basic inline code autocompletion