
#include <cum/Tensor.hpp>
#include <cum/Matrix.hpp>
#include <cum/Vector.hpp>
#include <iostream>
#include <fstream>
#include <string>

import yann;

int main()
{
    std::ifstream file("/home/abdul/datasets/sentences_pl.txt");
    std::string data; 
    std::string line;
    while (std::getline(file, line))
    {
        data += line;
    }

    yann::models::NGram model;
    model.fit(data);

    std::string token = "Dlatego";
    std::cout << token << ' ' << model.predict(token, 5) << '\n';

    return 0;
}