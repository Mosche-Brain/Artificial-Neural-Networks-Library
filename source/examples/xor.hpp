#include "Network/NeuralNet.hpp"
#include "NNVisualiser.hpp"

int xor_example()
{
    std::vector<Layer*> topology = 
    {
        new Layer(2, 1, pass, pass_prim, true),
        new Layer(3, 2, RELu, RELu_prim),
        new Layer(1, 3, sigmoid, sigmoid_prim)
    };
    
    std::cout << "meow\n";

    NeuralNet network(topology);
    NNVisualiser visualiser(&network);

    network.setLossFunction(cross_entropy);

    MatrixXd inputs {{0.0, 0.0},
                     {0.0, 1.0},
                     {1.0, 0.0},
                     {1.0, 1.0}};
    
    MatrixXd targets{{0},
                     {1},
                     {1},
                     {0}};
    
    for(int i = 0 ; i < inputs.rows() ; i++)
    {
        std::cout << '[' << inputs.row(i) << ']' << " -> " << network.forward(inputs.row(i)) << '\n';
    }
                    
    try
    {
        network.train(inputs, targets, 1000, 0.1);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::cout << "skibidi boop booop yes yes\n";
    }
    

    std::cout << ".\n";
    for(int i = 0 ; i < inputs.rows() ; i++)
    {
        std::cout << '[' << inputs.row(i) << ']' << " -> " << network.forward(inputs.row(i)) << '\n';
    }    

    visualiser.display();

    return 0;
}