#include <Eigen/Dense>

using namespace Eigen;

class VirtualNeuron
{
public:
    RowVectorXd* weights;
    double_t* bias;
    double_t* delta;
};