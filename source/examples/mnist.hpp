#include "Network/NeuralNet.hpp"
#include "containers/MNISTDataSet.hpp"

#include <thread>

void mnist_example()
{
    setNbThreads(std::thread::hardware_concurrency());

    int size = 28*28;
    std::vector<Layer*> topology =
    {
        new Layer(784, 1, pass, pass_prim, true),
        new Layer(256, 784, RELu, RELu_prim),
        new Layer(128, 256, RELu, RELu_prim),
        new Layer(10, 128, sigmoid, sigmoid_prim),
    };

    // std::vector<Layer*> topology =
    // {
    //     new Layer(784, 1, pass, pass_prim, true),
    //     new Layer(64, 784, RELu, RELu_prim),
    //     new Layer(16, 64, RELu, RELu_prim),
    //     new Layer(10, 16, sigmoid, sigmoid_prim),
    // };

    NeuralNet network(topology);
    network.setLossFunction(cross_entropy);

    MNISTDataSet dataset;
    // dataset.loadImages("./Data/MNIST/train-images-idx3-ubyte");
    // dataset.loadLabels("./Data/MNIST/train-labels-idx1-ubyte");
    dataset.loadImages("./Data/MNIST/t10k-images.idx3-ubyte");
    dataset.loadLabels("./Data/MNIST/t10k-labels.idx1-ubyte");


    MatrixXd onehot_labels = MatrixXd::Zero(dataset.labels().size(), 10);
    for(int i = 0 ; i < dataset.labels().size() ; i++)
    {
        // onehot_labels(i, dataset.labels()[i]) = 1.0;
        onehot_labels.row(i)[dataset.labels()[i]] = 1.0;
    }

    MatrixXd images_norm = dataset.images() / dataset.images().maxCoeff();

    std::cout << dataset.images().rows() << '\n';

    network.train(images_norm, onehot_labels, 50, 0.1);


    //network.export_to_json("weights-10k.json");
    

    // NNVisualiser visualiser(&network);

    // visualiser.display();

}