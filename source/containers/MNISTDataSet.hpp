#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

class MNISTDataSet 
{
public:
    MNISTDataSet();

    Eigen::VectorXd getImage(int index);
    double          getLabel(int index);

    Eigen::MatrixXd& images();
    Eigen::VectorXd& labels();

    void loadImages(const std::string& filepath);
    void loadLabels(const std::string& filepath);
    // Metody do ładowania obrazów i etykiet
    //static Eigen::MatrixXd loadImages(const std::string& filename);
    //static Eigen::VectorXd loadLabels(const std::string& filename);

private:
    Eigen::MatrixXd __images__;
    Eigen::VectorXd __labels__;

    int reverse_int(int integer);
};