# YANN - Yet Another Neural Networks library

A lightweight, modular minimalistic and easy to use C++ library for machine learning

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Czuowuek-SOS/Artificial-Neural-Networks-Library/graphs/commit-activity)

## 📜 Content

* [installation](#️-installaction-and-building-process)
* [examples of usage](#-examples-of-usage)
* [features](#-features)
* [credits](#-credits-and-used-technologies)
* [other](#-other-useless-informations)

## ❗ Disclaimers
* I used very poor english due to prove it wasn't written by LLM or something simillar 

## ⬇️ Installaction and building process

### requirements

* CMake
* OpenMP
* eigen3 or oneAPI or openBLAS
* G++ compiler or clang++
* CPU with at least one core

### 🐧 Linux and GNU/Linux

```bash
git clone https://github.com/Czuowuek-SOS/Artificial-Neural-Networks-Library
mv Artificial-Neural-Networks-Library Yet-Another-Artificial-Neural-Networks-Library
cd Yet-Another-Artificial-Neural-Networks-Library
mkdir build
cd build
cmake .. [options]
cmake --build . [options]
```

### 🪟 Windows and MacOS

Probably almost, just like in linux and GNU/Linux (I guess).

## 💡 Examples of usage

### Building project

```cmake 
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(gpt69)

add_executable(${project_name} main.cpp)
target_link_libraries(${project_name} PRIVATE YANN OpenMP::OpenMP_CXX cum::Core)

if(BUILD_USE_MKL)
    target_link_libraries(${project_name} PRIVATE cum::MKL cum_MKL)
    target_compile_definitions(${project_name} PRIVATE fsycl fPIC)
    target_compile_options(${project_name} PRIVATE -fsycl -qopenmp)
else()
    target_compile_options(${project_name} PRIVATE -fopenmp)
endif()
```

### Initializing library
#### Notes
* If you use different type than `FP32` must define used numeric type and backend before including yann or cum headers for proper compilation or add these definitions as a compile definitions in your CMakeLists.txt or compiler args.
* Before creating any Yann or cum objects you must call `cum::cum(cum::CUM_DEVICE device)` and pass `AUTO`/`CPU`/`GPU` next to `cum::CUM_DEVICE` due to device your want to use.
* I din't tested it yet on computers with more than one GPU (integrated or discrete) so it can select wrong one.
* Supported numeric types
    * `CUM_USE_F64`
    * `CUM_USE_F32` (defining it is optional)
    * `CUM_USE_F16`
    * `CUM_USE_BF16` (it might not compile properly)
    * `CUM_USE_Q8` (didn't tested yet )
* Currently supported backends
    * `BUILD_USE_MKL`
```cpp
#include <cum/cum.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::AUTO);
    
    /* you can put there some strange code */
    
    cum::decum()
}
```


### Creating and fitting sequential model
* In constructor of `yann::models::Sequential` class your can put initializer list filled with fabric methods of various layers types. 
```cpp
#include <yann/models/Sequential.hpp>

using namespace yann::models;

int main()
{
    int input_layer_size = 2;
    int hidden_layer_size = 69;
    int outut_layer_size = 1;
    Sequential model({layers::Input::createUnique(input_layer_size),
             layers::Dense::createUnique(hidden_layer_size, "relu"),
             layers::Dense::createUnique(outut_layer_size, "sigmoid")}); 

    int number_of_samples = 4;
    cum::Matrix x_train(number_of_samples, input_layer_size);
    cum::Matrix y_train(number_of_samples, input_layer_size);

    /* Fill training data here */

    cum::cumeric_t learning_rate = 0.1_c;
    int epochs = 200;

    // old way - not recommended
    // model.fit(x_train, y_train, learning_rate, epochs); 
    
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(learning_rate);
    
    model.fit(x_train, y_train, *optimizer, epochs);
    
    return 0;
}
```

### Creating and fitting perceptron

```cpp
#include <yann/models/Perceptron.hpp>

using namespace yann::models;

int main()
{
    int perceptron_input_size = 2;
    Perceptron model(perceptron_input_size, "relu");

    int number_of_samples = 2137;
    cum::Matrix x_train(number_of_samples, perceptron_input_size);
    cum::Vector y_train(number_of_samples);

    /* Filling training data */

    cum::cumeric_t learning_rate = 0.1_c;
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

cum::Matrix MyLayer::forward(const cum::Matrix& input)
{
    /* your implementation of forward pass */
}

cum::Matrix MyLayer::backward(const cum::Matrix deltaOutput) 
{
    /* your implementation of backward pass */
}

void MyLayer::update_weights(cum::cumeric_t rate) // This method is obsolete, now we are using dedicated optimizer class
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
* ✅ Working backward pass (Yes, this is insane)
* ✅ Compile time precision selections
* ✅ FP16 support
* ✅ Runtime device selection
* ✅ Multicore CPU acceleration
* ✅ GPU acceleration
* ✅ Templates-free math framework classes
* ✅ OneAPI support
* ⚠️ [Dedicated graphical envionment](https://github.com/Czuowuek-SOS/MLStudio) (work in progress)
* ❌ CUDA support
* ❌ ROCm support
* ❌ BF16, Q8 support
* ❌ Dynamic computational graphs
* ❌ Fully working Python binding
* ❌ Recurrent Neural Networks
* ❌ Transformers
* ❌ Outperform TensorFlow
* ❌ Built in telemetry

## 🧷 Credits and used technologies

* [OpenMP](https://www.openmp.org/) - Multiprocessing interface for C/C++
* [Eigen3](https://eigen.tuxfamily.org/) - C++ Linear Algebra library
* [oneAPI](https://oneapi.io/) - Open platform for heterogenous computing
* [Nlohmann JSON](https://github.com/nlohmann/json) - C++ library for parsing JSON format
* [Easy3D](https://github.com/LiangliangNan/Easy3D) - 3D visualizations library used in examples
* [Sarvel](https://sarvel.xyz/) - Literally Digital God

## 🥱 Other useless informations

### 🧼 Generative AI assistance in this project

* Name refactoring tasks
* Generating boilerplate
* Basic inline code autocompletion from free github copilot credits
* I also spend 0.05$ for Grok API tokens