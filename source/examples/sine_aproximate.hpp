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
    NeuralNet network({ Layers::Linear(1),
                        Layers::Tanh(256),
                        Layers::Tanh(256),
                        Layers::Tanh(256),
                        Layers::Tanh(1) });

        
    VectorXd input = VectorXd::LinSpaced(200, -4 * M_PI, 4 * M_PI);
    VectorXd target = input.unaryExpr([](double x) { return std::sin(x); });



    network.train(input, target, 800, 0.02);
    
    VectorXd results = network.predict(input);

    std::vector<double> X(input.data(), input.data() + input.size());
    std::vector<double> Y1(target.data(), target.data() + target.size());
    std::vector<double> Y2(results.data(), results.data() + results.size());


    matplot::plot(X, Y1, "-o ", X, Y2);
    matplot::show();

    return 0;
}