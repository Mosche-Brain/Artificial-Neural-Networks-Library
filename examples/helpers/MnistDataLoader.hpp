/*
 * @author: jaro
 * @name:   MnistDataLoader
 * @file:   examples/helpers/MnistDataLoader.hpp
 * @date:   18 October 2026 10:31:24
 */


#include <cum/Tensor.hpp>
#include <algorithm>
#include <print>
#include <cstdint>
#include <zlib.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <numeric>
#include <array>


#ifndef YANN_MNISTDATALOADER_HPP
#define YANN_MNISTDATALOADER_HPP

namespace helpers
{

class MnistDataLoader
{
public:
    struct Dataset
    {
        cum::Tensor images; // samples x 784, normalized to [0, 1]
        cum::Tensor labels; // samples x 10, one-hot encoded
    };

    static Dataset load(const std::string& directory, const std::string& split)
    {
        const std::string image_name =
            split == "train" ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte";
        const std::string label_name =
            split == "train" ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte";

        const std::string images_path = findFile(directory, image_name);
        const std::string labels_path = findFile(directory, label_name);

        gzFile images_file = gzopen(images_path.c_str(), "rb");
        gzFile labels_file = gzopen(labels_path.c_str(), "rb");
        if (images_file == nullptr)
            throw std::runtime_error("Cannot open MNIST images file: " + images_path);
        if (labels_file == nullptr)
        {
            gzclose(images_file);
            throw std::runtime_error("Cannot open MNIST labels file: " + labels_path);
        }

        try
        {
            const std::uint32_t image_magic = readU32(images_file, images_path);
            const std::uint32_t image_count = readU32(images_file, images_path);
            const std::uint32_t image_rows = readU32(images_file, images_path);
            const std::uint32_t image_cols = readU32(images_file, images_path);

            if (image_magic != 2051 || image_rows != 28 || image_cols != 28)
                throw std::runtime_error("Invalid MNIST image IDX header: " + images_path);

            const std::uint32_t label_magic = readU32(labels_file, labels_path);
            const std::uint32_t label_count = readU32(labels_file, labels_path);
            if (label_magic != 2049 || label_count != image_count)
                throw std::runtime_error("Invalid MNIST label IDX header: " + labels_path);

            const std::size_t image_size =
                static_cast<std::size_t>(image_rows) * image_cols;
            std::vector<cum::cumeric_t> image_data(
                static_cast<std::size_t>(image_count) * image_size);
            std::vector<cum::cumeric_t> label_data(
                static_cast<std::size_t>(label_count) * 10, 0.0f);

            std::vector<std::uint8_t> pixels(image_data.size());
            readBytes(images_file, pixels.data(), pixels.size(), images_path);
            for (std::size_t sample = 0; sample < image_count; ++sample)
            {
                for (std::size_t feature = 0; feature < image_size; ++feature)
                {
                    image_data[sample * image_size + feature] =
                        static_cast<cum::cumeric_t>(
                            pixels[sample * image_size + feature]) / 255.0f;
                }
            }

            std::vector<std::uint8_t> labels(label_count);
            readBytes(labels_file, labels.data(), labels.size(), labels_path);
            for (std::size_t sample = 0; sample < labels.size(); ++sample)
            {
                if (labels[sample] >= 10)
                    throw std::runtime_error("MNIST label is outside [0, 9]");
                label_data[sample * 10 + labels[sample]] = 1.0f;
            }

            gzclose(images_file);
            gzclose(labels_file);
            return
            {
                cum::Tensor::copy_memory({image_count, static_cast<cum::dim_t>(image_size)}, image_data.data()),
                cum::Tensor::copy_memory({label_count, 10}, label_data.data())
            };
        }
        catch (...)
        {
            gzclose(images_file);
            gzclose(labels_file);
            throw;
        }
    }

private:
    static std::string findFile(
        const std::string& directory,
        const std::string& name)
    {
        const std::string dotted =
            name.find("-idx") == std::string::npos
                ? name
                : name.substr(0, name.find("-idx")) + ".idx" +
                  name.substr(name.find("-idx") + 4);

        const std::vector<std::string> candidates = {
            directory + "/" + name,
            directory + "/" + name + ".gz",
            directory + "/" + dotted,
            directory + "/" + dotted + ".gz"
        };

        for (const std::string& path : candidates)
        {
            if (!std::filesystem::is_regular_file(path))
                continue;

            gzFile file = gzopen(path.c_str(), "rb");
            if (file != nullptr)
            {
                gzclose(file);
                return path;
            }
        }

        throw std::runtime_error("Cannot find MNIST IDX file for: " + name);
    }
    static std::uint32_t readU32(gzFile file, const std::string& path)
    {
        std::uint8_t bytes[4]{};
        readBytes(file, bytes, sizeof(bytes), path);
        return (static_cast<std::uint32_t>(bytes[0]) << 24) |
               (static_cast<std::uint32_t>(bytes[1]) << 16) |
               (static_cast<std::uint32_t>(bytes[2]) << 8) |
               static_cast<std::uint32_t>(bytes[3]);
    }

    static void readBytes(
        gzFile file,
        void* destination,
        std::size_t size,
        const std::string& path)
    {
        auto* bytes = static_cast<unsigned char*>(destination);
        std::size_t read = 0;
        while (read < size)
        {
            const unsigned int chunk = static_cast<unsigned int>(
                std::min<std::size_t>(size - read, 1u << 20));
            const int count = gzread(file, bytes + read, chunk);
            if (count <= 0)
                throw std::runtime_error("Unexpected end of MNIST IDX file: " + path);
            read += static_cast<std::size_t>(count);
        }
    }
};
} // helpers

#endif //YANN_MNISTDATALOADER_HPP
