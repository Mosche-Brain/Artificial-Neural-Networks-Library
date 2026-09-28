![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Czuowuek-SOS/Artificial-Neural-Networks-Library/graphs/commit-activity)
# Yet Another Neural Networks library

A lightweight, modular minimalistic and easy to use C++ library for machine learning

## 📜 Content

* [installation](#️-installaction-and-building-process)
* [examples of usage](#-examples-of-usage)
* [features](#-features)
* [credits](#-credits-and-used-technologies)
* [other](#-other-useless-informations)

## ❗ Disclaimers
* I used very poor english due to sleep quality.
* Library is currently under active development, there is no production ready realase (first is planned for 7 oct 2026)

# Architecture

## ⬇️ Installaction and building process

#### Requirements

* CMake
* oneAPI/OpenMP/~~cuda~~/~~ROCm~~/~~openBLAS~~
* G++, clang++ or Intel DPC++ compiler for oneAPI backend
* CPU with at least one core
* ~~At least one sexual crime in lifetime~~

#### Notes
* I deleted this section due to the latest changes, I must rewrite significant part of this document.

### 🐧 Linux and GNU/Linux
```bash
git clone https://github.com/Mosche-Brain/Artificial-Neural-Networks-Library
mv Artificial-Neural-Networks-Library Yet-Another-Artificial-Neural-Networks-Library
cd Yet-Another-Artificial-Neural-Networks-Library
mkdir build
cmake -B build [options]
cmake --build build
```

FP16 and oneAPI backend
```bash
cmake -B build -DCUM_USE_MKL=ON -DCUM_USE_FP16
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

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE YANN cum)

if(BUILD_USE_MKL)
    target_link_libraries(${PROJECT_NAME} PRIVATE cum::MKL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE fsycl fPIC)
    target_compile_options(${PROJECT_NAME} PRIVATE -fsycl -qopenmp)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
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
    
    cum::decum(); // Putting this isn't necessary
}
```

### Manipulating tensors

```cpp
#include <cum/Tensor.hpp>
#include <cum/cum.hpp>

int main()
{
    cum::Tensor A({6, 9, 7} cum::datatype::BF16, cum::layout::BSC);
    cum::Tensor B({6, 9, 7} cum::datatype::BF16, cum::layout::BSC);

    A.fill(67._c);
    B.fill(0.2137_c);
    
    cum::Tensor C = A * B; // matmul on the last two dims
    
    cum::cumeric_t scalar = C.at({6, 7, 1});
    
    B = A.multiply(B) // elementwise multiplication
    
    A = B.square(); // elementwise x^2
    
    // You can try to print the results, it really works

    return 67;
}
```

### Creating and fitting sequential model
* In constructor of `yann::models::Sequential` class your can put initializer list filled with fabric methods of various layers types.
```cpp
#include <print>

#include <yann/models/Sequential.hpp>

using namespace yann::models;

int main()
{
    cum::cum(cum::DEVICE::GPU); // In this case CPU will be much faster

    Sequential model({
        layers::Input::createUnique(2),
        layers::Dense::createUnique(3, "tanh"),
        layers::Dense::createUnique(outut_layer_size, "sigmoid")
    }); 

    cum::Tensor X(2, 4);
    X.at({0, 0}) = 0; X.at({1, 0}) = 0;;
    X.at({0, 1}) = 0; X.at({1, 1}) = 1;
    X.at({0, 2}) = 1; X.at({1, 2}) = 0;
    X.at({0, 3}) = 1; X.at({1, 3}) = 1;

    cum::Tensor Y(1, 4);
    Y.at({0, 0}) = 0;
    Y.at({0, 1}) = 1;
    Y.at({0, 2}) = 1;
    Y.at({0, 3}) = 0;

    cum::cumeric_t learning_rate = 0.1_c;
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.1);
    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();

    yann::runtime_config::set_verbosity(1);

    cum::Tensor Y_pred = model.forward(X);

    for (int i = 0 ; i < X.cols() ; i++)
        std::println("[{}, {}] -> {}", X(0, i), X(1, i), Y_pred(0, i));

    cum::dim_t epochs = 500;
    cum::dim_t batch = 4;
    model.fit(X, Y, *loss, *optimizer, epochs, batch);
    std::println("---------------------------------------");
    cum::Tensor Y_pred_2 = model.forward(X);
    
    for (int i = 0 ; i < X.cols() ; i++)
        std::println("[{}, {}] -> {}", X(0, i), X(1, i), Y_pred_2(0, i));

    return 0;
}
```

### Creating and fitting perceptron

```cpp
#include <yann/models/Perceptron.hpp>

using namespace yann::models;

int main()
{
    cum::dim_t perceptron_input_size = 2;
    Perceptron model(perceptron_input_size, "gelu");

    cum::dim_t number_of_samples = 2137;
    cum::Matrix x_train(number_of_samples, perceptron_input_size);
    cum::Vector y_train(number_of_samples);

    /* Filling training data */

    cum::cumeric_t learning_rate = 0.1_c;
    cum::dim_t epochs = 200;

    model.fit(x_train, y_train, learning_rate, epochs);
    return 0;
}
```

~~You can find full API documentation [there (currently not avaible)](www.amogus.org)~~

Full documentation will be avaible [here](brain.mosche.dev/docs)

## 🔨 Features

* ✅ Sequential neural networks
* ✅ Dense layers
* ✅ Perceptrons
* ✅ Reasonable design
* ✅ Working backward pass (Yes, this is insane)
* ✅ FP16 support
* ✅ Callbacks system
* ✅ Runtime device selection
* ✅ Multicore CPU acceleration
* ✅ GPU acceleration
* ✅ Whole math framework
* ✅ OneAPI support
* ✅ Compiling code 
* ✅ ND Tensors
* ⚠️ BF16 support (probably works now)
* ⚠️ int8&uint8 support (probably)
* ⚠️ Reasonable unit tests (maybe)
* ⚠️ Fused kernels for neural networks
* ⚠️ Dynamic computational graphs
* ⚠️ Convolutional layers
* ⚠️ Compile time code traces pruning
* ⚠️ [Dedicated graphical envionment](https://github.com/Mosche-Brain/MLStudio) (work in progress)
* ❌ CUDA support
* ❌ ROCm support
* ❌ Embbeded version (for xtensa or arm bare metal)
* ❌ [Python binding](https://pypi.org/project/cumpy/0.1.0/)
* ❌ C binding 
* ❌ FP8
* ❌ Safetensors format
* ❌ Recurrent Neural Networks
* ❌ Pooling layers
* ❌ Normalization layers
* ❌ Hoppfield networks
* ❌ SVM's
* ❌ RL Stuff
* ❌ Transformers
* ❌ Wielogłowicowa uwaga
* ❌ Built in telemetry

## 🧷 Credits and used technologies

* [oneAPI](https://oneapi.io/) - Open platform for heterogenous computing
* [OpenMP](https://www.openmp.org/) - Multiprocessing interface for C/C++
* [Eigen3](https://eigen.tuxfamily.org/) - C++ Linear Algebra library used in early stage of development
* [Nlohmann JSON](https://github.com/nlohmann/json) - C++ library for parsing JSON format
* [Easy3D](https://github.com/LiangliangNan/Easy3D) - 3D visualizations library used in examples
* [safetensors.cpp](https://github.com/carsonpo/safetensors.cpp) - Base for my implementation of Hugging Face safetensors format
* [Sarvel](https://sarvel.xyz/) - Literally Digital God

## 🥱 Other useless informations

### 💻 Hardware used during development (this may not work on anything else)

* Intel Core Ultra 5 250k plus
* Intel Arc B580
* GX-420GI Radeon R7E

### 🧼 Generative AI assistance in this project

* Name refactoring tasks
* Generating boilerplate
* Basic inline code autocompletion from free github copilot credits
* I also spend ~~0.05$~~ 2.67\$ for Grok and Kimi tokens
* Testing Nemotron 3.5 Lightning 30B A3B for generating doxygen