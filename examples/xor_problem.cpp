#include <cum/cum.hpp>
#include <matplot/matplot.h>

#include <vector>
#include <iostream>
#include <algorithm>

std::vector<std::vector<float>> make_matrix_row_major(const cum::Matrix& mat)
{
    std::size_t rows = mat.rows();
    std::size_t cols = mat.cols();

    std::vector<std::vector<float>> M(rows, std::vector<float>(cols));

    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            M[r][c] = static_cast<float>(mat(r, c)); // sycl::half -> float
        }
    }

    return M;
}

std::vector<std::vector<float>> make_min_2x2(const std::vector<std::vector<float>>& M)
{
    std::size_t rows = M.size();
    std::size_t cols = M.empty() ? 0 : M[0].size();

    std::size_t out_rows = std::max<std::size_t>(rows, 2);
    std::size_t out_cols = std::max<std::size_t>(cols, 2);

    std::vector<std::vector<float>> out(out_rows, std::vector<float>(out_cols));

    for (std::size_t r = 0; r < out_rows; ++r)
    {
        for (std::size_t c = 0; c < out_cols; ++c)
        {
            std::size_t rr = std::min(r, rows - 1);
            std::size_t cc = std::min(c, cols - 1);
            out[r][c] = M[rr][cc];
        }
    }

    return out;
}

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);

    cum::Matrix weights_hidden = cum::Matrix::Random(3, 2);
    cum::Matrix weights_output = cum::Matrix::Random(3, 1);

    auto W_h = make_min_2x2(make_matrix_row_major(weights_hidden));
    auto W_o = make_min_2x2(make_matrix_row_major(weights_output));

    matplot::figure(true);

    matplot::subplot(1, 2, 1);
    matplot::imagesc(W_o);
    matplot::axis(matplot::equal);
    matplot::title("Output weights");

    matplot::subplot(1, 2, 2);
    matplot::imagesc(W_h);
    matplot::axis(matplot::equal);
    matplot::title("Hidden weights");

    // matplot::colorbar();
    matplot::show();

    cum::decum();

    return 0;
}