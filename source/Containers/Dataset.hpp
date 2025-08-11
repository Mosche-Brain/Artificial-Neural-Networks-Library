#include <Eigen/Dense>
#include <functional>

class Dataset
{
public:
    void loadFromFile();

    void normalize();
    void transform(std::function<double(double)> f);

    Eigen::MatrixXd get_train_x();
    Eigen::VectorXd get_train_y();

protected:
    Eigen::MatrixXd train_x;
    Eigen::VectorXd train_y;
};