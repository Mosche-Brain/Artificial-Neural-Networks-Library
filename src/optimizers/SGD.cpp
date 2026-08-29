//
// Created by jaro on 7/17/26.
//

#include "optimizers/SGD.hpp"

#include "cum/runtime.hpp"

#include <fstream>
#include <string>

namespace
{
    void dump_matrix(
        const cum::Matrix& matrix,
        const char* name,
        std::size_t step,
        std::size_t parameter)
    {
        std::ofstream file(
            "plots/sgd_" + std::to_string(step) + "_parameter_" +
            std::to_string(parameter) + "_" + name + ".txt");
        file << "# rows " << matrix.rows() << " cols " << matrix.cols() << '\n';
        for (std::size_t row = 0; row < matrix.rows(); ++row)
        {
            for (std::size_t col = 0; col < matrix.cols(); ++col)
            {
                if (col != 0)
                {
                    file << ' ';
                }
                file << static_cast<double>(matrix(row, col));
            }
            file << '\n';
        }
    }
}

namespace yann::optimizers
{
    SGD::SGD(cum::cumeric_t learning_rate)
        : OptimizerBase(learning_rate)
    {
    }

    void SGD::step(cum::Matrix& params, cum::Matrix& grad)
    {
    }

    void SGD::step(std::vector<Parameter*>& params)
    {
        static std::size_t update_step = 0;
        for (std::size_t parameter = 0; parameter < params.size(); ++parameter)
        {
            Parameter* param = params[parameter];
            const std::size_t step = update_step++;
            if (step < 16)
            {
                std::ofstream metadata(
                    "plots/sgd_" + std::to_string(step) + "_parameter_" +
                    std::to_string(parameter) + "_metadata.txt");
                metadata << "learning_rate "
                         << static_cast<double>(learning_rate) << '\n';
                dump_matrix(param->values, "values_before", step, parameter);
                dump_matrix(param->gradient, "gradient", step, parameter);
            }

            param->values -= param->gradient * learning_rate;
            cum::runtime::sync();

            if (step < 16)
            {
                dump_matrix(param->values, "values_after", step, parameter);
            }

            param->clear_gradient();
            cum::runtime::sync();
        }
    }

    std::unique_ptr<SGD> SGD::create(cum::cumeric_t learning_rate)
    {
        return std::make_unique<SGD>(learning_rate);
    }
} // namespace yann::optimizers
