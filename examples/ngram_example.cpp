#include <ANN/Models/NGram.hpp>

#include <iostream>

int main()
{
    std::ifstream file("/home/abdul/datasets/sentences_pl.txt");
    std::string data; 
    std::string line;
    while (std::getline(file, line))
    {
        data += line;
    }



    YANN::Models::NGram model;
    model.fit(data);

    std::string token = "Dlatego";
    std::cout << token << ' ' << model.predict(token, 5) << '\n';

    return 0;
}