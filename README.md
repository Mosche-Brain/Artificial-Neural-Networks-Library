# YANN - Yet Another Neural Networks library

A lightweight, modular minimalistic and easy to use C++ library for machine learning

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Czuowuek-SOS/Artificial-Neural-Networks-Library/graphs/commit-activity)

## Table of content

* [installation](#️-installaction-and-building-process)
* [examples of usage](#-examples-of-usage)
* [features](#-features)
* [credits](#-credits-and-used-technologies)
* [other](#-other-useless-informations)

## ⬇️ Installaction and building process

### requirements

* eigen3 devel
* OpenMP
* G++ compiler
* CMake
* CPU with at least one core

### 🐧 Linux and GNU/Linux

```bash
git clone https://github.com/Czuowuek-SOS/Artificial-Neural-Networks-Library
cd Artificial-Neural-Networks-Library
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
    int outut_layer_size = 1;
    Sequential model({Layers::Input::createUnique(input_layer_size),
             Layers::Dense::createUnique(hidden_layer_size, "relu"),
             Layers::Dense::createUnique(outut_layer_size, "sigmoid")}); 
                      /* you can also put tanh, gelu, softmax, etc */

    int number_of_samples = 4;
    YANN::matrix_t x_train(number_of_samples, input_layer_size);
    YANN::matrix_t y_train(number_of_samples, input_layer_size);

    /* Filling training data */
    YANN::numeric_t learning_rate = 0.1;
    int epochs = 200;

    model.fit(x_train, y_train, learning_rate, epochs);
    return 0;
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

    int number_of_samples = 2137;
    YANN::matrix_t x_train(number_of_samples, perceptron_input_size);
    YANN::vector_t y_train(number_of_samples);

    /* Filling training data */
    YANN::numeric_t learning_rate = 0.1;
    int epochs = 200;

    model.fit(x_train, y_train, learning_rate, epochs);
    return 0;
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

You can find full API documentation [there (currently not avaible)](www.amogus.org)

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
* ❌ Computational graphs
* ❌ Python API
* ❌ Recurent Neural Networks
* ❌ Transformers
* ❌ Outperform TensorFlow
* ❌ Built in telemetry

## 🧷 Credits and used technologies

* [Eigen3](https://eigen.tuxfamily.org/) - C++ Linear Algebra library
* [OpenMP](https://www.openmp.org/) - Multiprocessing interface for C/C++
* [Nlohmann JSON](https://github.com/nlohmann/json) - C++ library for parsing JSON
* [Sarvel](https://sarvel.xyz/) - Literally Digital God
* [Jakub Gładysz](https://sarvel.xyz/) - PhD at PWr

## 🥱 Other useless informations

### 🧼 Generative AI assistance in this project

* Name refactoring tasks
* Generating boilerplate
* Basic inline code autocompletion
