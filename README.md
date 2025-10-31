# Simple Neural Networks library


## Examples of usage

### Creating and fitting sequential model
```cpp
#include <SNN/Models/Sequential.hpp>

using namespace SNN::Models;

int main()
{
    Sequential model({Layers::Input::createUnique(2),
			          Layers::Dense::createUnique(3, "relu")
			          Layers::Dense::createUnique(1, "sigmoid")});

	Eigen::MatrixXf x_train(4, 2);
	Eigen::VectorXf y_train(4);
    /* Filling training data */

	model->fit(x_train, y_train, 0.1f, 50);
}
```

### Creating and fitting perceptron
```cpp
#include <SNN/Models/Perceptron.hpp>

using namespace SNN::Models;

int main()
{
    Perceptron model(4, "relu");

	Eigen::MatrixXf x_train(50, 4);
	Eigen::VectorXf y_train(50);
    /* Filling training data */

    model.fit(x_train, y_train,, 0.1f, 200);
}
```

### Using one hot encoding
```cpp
```

### Defining new layer types
```cpp
#include "MyLayer.hpp"

MyLayer::MyLayer(int layerSize)
{
    /* Initializing layer params */

    this->_initialized_ = true;
}

Eigen::MatrixXf MyLayer::forward(const Eigen::MatrixXf& input)
{
    this->outputs = /* ... */;
    return outputs;   
}

Eigen::MatrixXf MyLayer::backward(const Eigen::MatrixXf deltaOutput)
{
    /* your backpropagation implementation */
}

void MyLayer::update_weights(float_t rate)
{
    /* Your method to updating weights */
}

std::unique_ptr<LayerBase> MyLayer::createUnique(int layerSize)
{
    /* Layer class fabric method */
    return std::make_unique<MyLayer>(layerSize);
}

```

### Using RNNs
```cpp
//comming soon
```

### Using Spiking Neural Networks
```cpp
// comming soon
```

### Using Hoppfield Networks
```cpp
// comming soon
```