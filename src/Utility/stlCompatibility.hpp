#pragma once

#include <vector>
#include <list>

#include <Eigen/Dense>
#include <Eigen/Core>

/*  this code is stoled from here: 
    https://gist.github.com/ialhashim/14db82926cdab2575508 */

namespace ANN::Utils::stlCompatibility
{
    /*
    template<typename T>
    Eigen::VectorX<T> stdVector2Eigen(std::vector<T> vector);
    
    template<typename T>
    std::vector<T> Eigen2stdVector(Eigen::VectorX<T> vector);
    
    template<typename T>
    Eigen::VectorX<T> stdList2Eigen(std::vector<T> vector);
    
    template<typename T>
    std::vector<T> Eigen2stdList(Eigen::VectorX<T> vector);
    */

    template<typename Scalar, typename Container>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> toEigenMatrix(const Container& vectors)
    {
        if (vectors.empty()) {
            return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(0, 0);
        }

        size_t rows = vectors.size();
        size_t cols = vectors.front().size();

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> M(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            M.row(i) = Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>(vectors[i].data(), cols);
        }

        return M;
    }

    template<typename Scalar, typename Container>
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> toEigenVector(const Container& vector)
    {
        if (vector.empty()) {
            return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(0, 1);
        }

        size_t lenght = vector.size();

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> V(lenght);

        V = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(vector.data(), lenght);

        return V;
    }

    template<typename Matrix>
    std::vector<std::vector<typename Matrix::Scalar>> fromEigenMatrix(const Matrix& M)
    {
        std::vector<std::vector<typename Matrix::Scalar>> m(M.rows());
        
        for (int i = 0; i < M.rows(); ++i) {
            m[i].resize(M.cols());
            Eigen::Map<Eigen::Matrix<typename Matrix::Scalar, 1, Eigen::Dynamic>>(m[i].data(), M.cols()) = M.row(i);
        }
        
        return m;
    }

    template<typename Vector>
    std::vector<typename Vector::Scalar> fromEigenVector(const Vector& V)
    {
        std::vector<typename Vector::Scalar> v(V.size());

        Eigen::Map<Eigen::Matrix<typename Vector::Scalar, Eigen::Dynamic, 1>>(v.data(), V.size());

        return v;
    }

    template<typename Scalar>
    using STLMatrix = std::vector<std::vector<Scalar>>;
} 