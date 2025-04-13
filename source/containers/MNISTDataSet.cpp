#include "containers/MNISTDataSet.hpp"

Eigen::VectorXd MNISTDataSet::getImage(int index)
{
    return this->__images__.row(index).transpose();
}

double MNISTDataSet::getLabel(int index)
{
    return this->__labels__[index];
}

Eigen::MatrixXd* MNISTDataSet::images()
{
    return &this->__images__;
}

Eigen::VectorXd* MNISTDataSet::labels()
{
    return &this->__labels__;
}

void MNISTDataSet::loadImages(const std::string& filepath)
{
    std::ifstream filestream(filepath, std::ios::binary);
    if (!filestream.is_open())
    {
        throw std::runtime_error("Couldn't open file " + filepath);
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    filestream.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    
    filestream.read((char*)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);

    filestream.read((char*)&rows, sizeof(rows));
    rows = reverse_int(rows);
    
    filestream.read((char*)&cols, sizeof(cols));
    cols = reverse_int(cols);

    this->__images__.resize(num_images, rows * cols);
    std::vector<uint8_t> buffer(rows * cols);

    for (int i = 0; i < num_images; i++) 
    {
        filestream.read((char*)buffer.data(), rows * cols);
        for (int j = 0; j < rows * cols; j++)
        {
            this->__images__(i, j) = static_cast<double>(buffer[j]);
        }
    }
}

void MNISTDataSet::loadLabels(const std::string& filepath) 
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Couldn't open file " + filepath);
    }

    int magic_number = 0, num_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    
    file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    std::vector<uint8_t> buffer(num_labels);
    file.read((char*)buffer.data(), num_labels);

    this->__labels__.resize(num_labels);
    for (int i = 0; i < num_labels; i++) {
        this->__labels__[i] = static_cast<double>(buffer[i]);
    }
}

int MNISTDataSet::reverse_int(int integer) 
{
    return __builtin_bswap32(integer);
}