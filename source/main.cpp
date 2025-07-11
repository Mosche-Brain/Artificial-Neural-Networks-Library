#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <thread>
#include <cmath>

#include <nlohmann/json.hpp>

// #include "openGLPlotLive/plot/plot.h"
// #include "openGLPlotLive/window/window.h"
// #include "openGLPlotLive/lines/Line2DVecfVecGLMV3.h"
// #include "openGLPlotLive/scatterPlot/"
// [abdul@abdul-manjaro datasets]$ youtube-dl -f bestvideo+bestaudio --merge-output-format mp4 --postprocessor-args "-ss 00:18:56 -t 00:19:36" "https://www.youtube.com/watch?v=RsDoUUnV4ls"


#include "Network/Perceptron.hpp"
#include "Network/NeuralNet.hpp"
#include "containers/MNISTDataSet.hpp"
#include "NNVisualiser.hpp"

#include "examples/xor.hpp"

using json = nlohmann::json;

MatrixXd getTestSample(const MatrixXd& data, int sampleSize);
MatrixXd loadIrisData(const std::string& filename);
double computeAccuracy(const VectorXd& predictions, const VectorXd& expected);
void train(NeuralNet* net, MNISTDataSet& dataset, int epochs, double rate);

int main()
{
    NeuralNet network({ Layers::Linear(1),
                        Layers::Tanh(7),
                        Layers::Tanh(7),
                        Layers::Tanh(5),
                        Layers::Tanh(3),
                        Layers::Tanh(1) });

    // network.setLossFunction(MSE);
    // xor_example();


    //VectorXd input;
    int n = 200;
    MatrixXd input(n , 1);
    float max = 4*M_PI;
    float step = max / n * 2;
    float j = -max;
    for(int i = 0 ; i < n ; i++)
    {
        input(i, 0) = j;
        j += step;
    }
    
    std::cout << input(0, 0) << "\n";
    std::cout << input(199, 0) << "\n";
    std::cout << "brrrrr\n";
    VectorXd target(n);
    for(int i = 0 ; i < input.rows() ; i++)
    {
        // target << tanh(input[i]); // tanh activation function
        // target << input[i] * input[i]; // quadratic function
        // target << 1 / (1 + exp(-input[i])); // sigmoid activation function
        // target << input[i] > 0 ? 1 : 0; // step function
        // target << input[i] < 0 ? -1 : 1; // sign function
        target[i] = sinf(input.row(i)[0]); // sine function;
        // target[i] = input.row(i)[0] * 3;
    }
                       
    NNVisualiser visualiser(&network);
    
    visualiser.train_x = input;
    visualiser.train_y = target;
    
    visualiser.display();

    
    // network.train(input, target, 500, 0.1);

    return 0;
}

// void train(NeuralNet* net, MNISTDataSet& dataset, int epochs, double rate)
// {
//     for(int _ = 0 ; _ < epochs ; _++)
//     {
//         for(int i = 0 ; i < dataset.labels()->size() ; i++)
//         {
//             RowVectorXd image = dataset.getImage(i);
//             VectorXd expected = VectorXd::Zero(10); 
//             expected[dataset.getLabel(i)] = 1;

//             //VectorXd output = net->forward(image);
//             //net->backpropagate(expected, rate);
//         }
//     }
// }


// MatrixXd loadIrisData(const std::string& filename)
// {
//     std::ifstream file(filename);
//     if (!file.is_open()) 
//     {
//         throw std::runtime_error("Nie można otworzyć pliku " + filename);
//     }

//     json data;
//     file >> data;
//     file.close();

//     int rows = data.size();
//     int cols = 5; // 4 cechy (bez kolumny "class")

//     MatrixXd irisMatrix(rows, cols);

//     std::map<std::string, int> species =
//     {
//         {"setosa", 0},
//         {"versicolor", 1},
//         {"virginica", 2}
//     };

//     for (int i = 0; i < rows; ++i) 
//     {
//         irisMatrix(i, 0) = data[i]["sepalLength"];
//         irisMatrix(i, 1) = data[i]["sepalWidth"];
//         irisMatrix(i, 2) = data[i]["petalLength"];
//         irisMatrix(i, 3) = data[i]["petalWidth"];
//         irisMatrix(i, 4) = species[data[i]["species"]];
//     }

//     return irisMatrix;
// }


// MatrixXd getTestSample(const MatrixXd& data, int sampleSize)
// {
//     if (sampleSize > data.rows()) 
//     {
//         throw std::invalid_argument("Próbka testowa jest większa niż zbiór danych!");
//     }

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dist(0, data.rows() - 1);

//     MatrixXd testSample(sampleSize, data.cols());
//     for (int i = 0; i < sampleSize; ++i) 
//     {
//         int idx = dist(gen);
//         testSample.row(i) = data.row(idx);
//     }

//     return testSample;
// }

double computeAccuracy(const VectorXd& predictions, const VectorXd& expected)
{
    if (predictions.size() != expected.size()) 
    {
        throw std::invalid_argument("📌 Błąd: Rozmiary wektorów predictions i expected muszą być identyczne!");
    }

    int correct = (predictions.array() == expected.array()).count(); // Liczba poprawnych klasyfikacji
    double accuracy = (double)correct / predictions.size(); // Obliczenie skuteczności (procent poprawnych)

    return accuracy * 100.0; // Wynik w procentach
}