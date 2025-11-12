#include <ANN/Utility/functions.hpp>

#include <matplot/matplot.h>

int main() {

    // matplot::fplot("cos(x)", "o-r")->line_width(2);
    matplot::fplot([](float x) { return ANN::Utils::sigmoid(x);});
    matplot::hold(matplot::on);
    matplot::fplot([](float x) { return ANN::Utils::sigmoid_derivative(x);});

    matplot::show();
    return 0;
}
