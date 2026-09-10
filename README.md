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
* I used very poor english due to sleep quality.
* Library is currently under active development, there is no production ready realase (first is planned for 7 oct 2026)

## ⬇️ Installaction and building process

#### Requirements

* CMake
* oneAPI/OpenMP/~~cuda~~/~~ROCm~~/~~openBLAS~~
* G++, clang++ or Intel DPC++ compiler for oneAPI backend
* CPU with at least one core
* ~~At least one sexual crime in lifetime~~

#### Notes
* Supported numeric types
  `CUM_USE_F64` `CUM_USE_F32` `CUM_USE_F16` ~~`CUM_USE_BF16`~~ ~~`CUM_USE_Q8`~~
* Currently supported backends `BUILD_USE_MKL`
* BF16 and Q8 may not compile properly

### 🐧 Linux and GNU/Linux
```bash
git clone https://github.com/Czuowuek-SOS/Artificial-Neural-Networks-Library
mv Artificial-Neural-Networks-Library Yet-Another-Artificial-Neural-Networks-Library
cd Yet-Another-Artificial-Neural-Networks-Library
mkdir build
cmake -B build [options]
cmake --build build
```

FP16 and oneAPI backend
```bash
cmake -B build -DCUM_USE_MKL=ON -DCUM_USE_F16
```

### 🪟 Windows and MacOS

Probably almost, just like in linux and GNU/Linux (I guess).

## 💡 Getting started

* ⚠️ Some examples may not be accurate due to recent changes

### Building project
```cmake 
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(gpt69)

add_executable(${project_name} main.cpp)
target_link_libraries(${project_name} PRIVATE YANN OpenMP::OpenMP_CXX cum::Core)

if(BUILD_USE_MKL)
    target_link_libraries(${project_name} PRIVATE cum::MKL cumL)
    target_compile_definitions(${project_name} PRIVATE fsycl fPIC)
    target_compile_options(${project_name} PRIVATE -fsycl -qopenmp)
else()
    target_compile_options(${project_name} PRIVATE -fopenmp)
endif()
```

### Initializing library

* If you use different type than `FP32` you must define used numeric type and backend before including yann or cum headers for proper compilation or add these definitions as a compile definitions in your CMakeLists.txt or compiler args.
* Before creating any Yann or cum objects you must call `cum::cum(cum::DEVICE device)` and eventually pass `AUTO`/`CPU`/`GPU` next to `cum::CUM_DEVICE` due to device your want to use.
* I din't tested it yet on computers with more than one GPU (integrated or discrete) so it can select wrong one.

```cpp
#include <cum/cum.hpp>

int main()
{
    cum::cum(cum::DEVICE::AUTO);
    
    /* you can put there some strange code */
    
    cum::decum() // Putting this isn't necessary
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
    Sequential model({
        layers::Input::createUnique(input_layer_size),
        layers::Dense::createUnique(hidden_layer_size, "relu"),
        layers::Dense::createUnique(outut_layer_size, "atan")
    }); 

    int number_of_samples = 4;
    cum::Matrix x_train(number_of_samples, input_layer_size);
    cum::Matrix y_train(number_of_samples, input_layer_size);

    /* Fill training data here */

    cum::cumeric_t learning_rate = 0.1_c;
    cum::dim_t epochs = 200;

    yann::loss::Loss loss = yann::loss::MeanSquaredError::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(learning_rate);
    
    model.fit(x_train, y_train, *loss, *optimizer, epochs);
    
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
    Perceptron model(perceptron_input_size, "gelu");

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

~~You can find full API documentation [there (currently not avaible)](www.amogus.org)~~

Full documentation will be avaible [here](brain.mosche.dev/docs)

## 🔨 Features

* ✅ Sequential models
* ✅ Dense layers
* ✅ Polymorphic optimizers and loss functions
* ✅ Working backward pass (Yes, this is insane)
* ✅ Compile time precision selections
* ✅ FP16 support
* ✅ Runtime device selection
* ✅ Multicore CPU acceleration
* ✅ GPU acceleration
* ✅ Templates-free math framework
* ✅ OneAPI support
* ⚠️ N-dimensional Tensors
* ⚠️ Fused kernels for neural networks
* ⚠️ Fused kernels for neural networks
* ⚠️ Dynamic computational graphs
* ⚠️ Convolutional layers
* ⚠️ [Dedicated graphical envionment](https://github.com/Mosche-Brain/MLStudio) (work in progress)
* ❌ CUDA support
* ❌ ROCm support
* ❌ BF16, Q8 support
* ❌ Fully working Python binding
* ❌ Recurrent Neural Networks
* ❌ Transformers
* ❌ Built in telemetry

## 🧷 Credits and used technologies

* [oneAPI](https://oneapi.io/) - Open platform for heterogenous computing
* [OpenMP](https://www.openmp.org/) - Multiprocessing interface for C/C++
* [Eigen3](https://eigen.tuxfamily.org/) - C++ Linear Algebra library used in early stage of development
* [Nlohmann JSON](https://github.com/nlohmann/json) - C++ library for parsing JSON format
* [Easy3D](https://github.com/LiangliangNan/Easy3D) - 3D visualizations library used in examples
* [Sarvel](https://sarvel.xyz/) - Literally Digital God

## 🥱 Other useless informations

### 💻 Hardware used during development (this may not work on anything else)

* Intel Core Ultra 5 250k plus
* Intel Arc B580

### 🧼 Generative AI assistance in this project

* Name refactoring tasks
* Generating boilerplate
* Basic inline code autocompletion from free github copilot credits
* I also spend 0.05$ for Grok API tokens
