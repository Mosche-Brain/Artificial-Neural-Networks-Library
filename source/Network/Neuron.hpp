#include <Eigen/Dense>

using Eigen::RowVectorXd;


class VirtualNeuron
{
public:
    RowVectorXd* weights;
    double_t* bias;
    double_t* delta;
};