#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <thread>
#include <cmath>

#include <nlohmann/json.hpp>

#include <matplot/matplot.h>

// [abdul@abdul-manjaro datasets]$ youtube-dl -f bestvideo+bestaudio --merge-output-format mp4 --postprocessor-args "-ss 00:18:56 -t 00:19:36" "https://www.youtube.com/watch?v=RsDoUUnV4ls"


#include "Network/Perceptron.hpp"
#include "Network/NeuralNet.hpp"
#include "containers/MNISTDataSet.hpp"
#include "NNVisualiser.hpp"

#include "examples/xor.hpp"

using json = nlohmann::json;
// using plt = matplot; // matplot is a namespace, use matplot:: directly

MatrixXd getTestSample(const MatrixXd& data, int sampleSize);
MatrixXd loadIrisData(const std::string& filename);
double computeAccuracy(const VectorXd& predictions, const VectorXd& expected);
void train(NeuralNet* net, MNISTDataSet& dataset, int epochs, double rate);

int main()
{
    NeuralNet network({ Layers::Linear(2),
                        Layers::Tanh(256),
                        Layers::Tanh(1),
                        Layers::Linear(1) });


    // Matrix<double, 300, 2> inputs(300, 2);
    // MatrixXd inputs(300, 2) = MatrixXd::;
    MatrixXd inputs = MatrixXd::Random(300, 2);
    
    
    auto f = [&network](double x, double y) -> double
    {
        Vector2d vec(2);
        vec << x, y;
        
        // return network.forward(vec).sum();
        return std::sin(std::sqrt(std::pow(vec.x() + vec.y(), 2)));
    };
    
    MatrixXd targets(300, 1);
    for(int i = 0 ; i < inputs.rows() ; i++)
    {
        // targets.row(i)[0] = inputs.row(i).sum();
        targets.row(i)[0] = f(inputs.row(i)[0], inputs.row(i)[1]);
    }    

    network.train(inputs, targets, 500, 0.05);

    // VectorXd results = network.predict(inputs);

    auto n = [&network](double x, double y) -> double
    {
        Vector2d vec(2);
        vec << x, y;

        return network.forward(vec)[0];
    };


    // matplot::fsurf(f);
    matplot::fmesh(n);

    matplot::xlabel("x");
    matplot::ylabel("y");
    matplot::zlabel("z");
    matplot::view(90, 30, 30);


    matplot::show();

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