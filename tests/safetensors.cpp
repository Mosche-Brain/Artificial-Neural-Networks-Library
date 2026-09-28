/*
 * @author: jaro
 * @name:   safetensors
 * @file:   tests/safetensors.cpp
 * @date:   28 September 2026 14:35:30
 */


#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include <string>

#include <cum/cum.hpp>

#include "yann/utils/safetensors.hpp"

namespace
{
    struct TempFile
    {
        std::filesystem::path path;

        explicit TempFile(const char* name)
            : path(std::filesystem::temp_directory_path() / name)
        {
        }

        ~TempFile()
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    };

    cum::Tensor make_vector()
    {
        cum::Tensor tensor(cum::Shape{6}, cum::datatype::FP32);

        tensor.at<float>({0}) = 1.0f;
        tensor.at<float>({1}) = -2.0f;
        tensor.at<float>({2}) = 3.5f;
        tensor.at<float>({3}) = 0.0f;
        tensor.at<float>({4}) = 42.0f;
        tensor.at<float>({5}) = -123.25f;

        return tensor;
    }

    cum::Tensor make_matrix()
    {
        cum::Tensor tensor(cum::Shape{2, 3}, cum::datatype::FP32);

        tensor.at<float>({0, 0}) = 1.0f;
        tensor.at<float>({0, 1}) = 2.0f;
        tensor.at<float>({0, 2}) = 3.0f;

        tensor.at<float>({1, 0}) = -4.0f;
        tensor.at<float>({1, 1}) = -5.0f;
        tensor.at<float>({1, 2}) = -6.0f;

        return tensor;
    }

    cum::Tensor make_3d_tensor()
    {
        cum::Tensor tensor(cum::Shape{2, 2, 2}, cum::datatype::FP32);

        tensor.at<float>({0, 0, 0}) = 1.0f;
        tensor.at<float>({0, 0, 1}) = 2.0f;
        tensor.at<float>({0, 1, 0}) = 3.0f;
        tensor.at<float>({0, 1, 1}) = 4.0f;

        tensor.at<float>({1, 0, 0}) = 5.0f;
        tensor.at<float>({1, 0, 1}) = 6.0f;
        tensor.at<float>({1, 1, 0}) = 7.0f;
        tensor.at<float>({1, 1, 1}) = 8.0f;

        return tensor;
    }

    // void require_same_tensor(cum::Tensor& expected, cum::Tensor& actual)
    // {
    //     REQUIRE(actual.type() == expected.type());
    //     REQUIRE(actual.shape() == expected.shape());
    //     REQUIRE(actual.size() == expected.size());
    //     REQUIRE(actual == expected);
    // }

    void require_same_tensor(const cum::Tensor& expected, const cum::Tensor& actual)
    {
        REQUIRE(actual.type() == expected.type());
        REQUIRE(actual.shape() == expected.shape());
        REQUIRE(actual.size() == expected.size());

        const auto shape = expected.shape();

        if (shape.size() == 1)
        {
            for (std::int64_t i = 0; i < shape[0]; ++i)
            {
                INFO("index = " << i);
                REQUIRE(actual.at<float>({i}) == expected.at<float>({i}));
            }
        }

        REQUIRE(actual == expected);
    }

}

TEST_CASE("Safetensors: save and load a vector")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_vector.safetensors");

    auto original = make_vector();

    yann::utils::save_safetensors(
        {{"tensor", original}},
        file.path.string()
    );

    auto loaded = yann::utils::load_safetensors(file.path.string());

    REQUIRE(loaded.size() == 1);
    REQUIRE(loaded.contains("tensor"));

    auto& result = loaded.at("tensor");

    require_same_tensor(original, result);
}

TEST_CASE("Safetensors: save and load a matrix")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_matrix.safetensors");

    auto original = make_matrix();

    yann::utils::save_safetensors(
        {{"matrix", original}},
        file.path.string()
    );

    auto loaded = yann::utils::load_safetensors(file.path.string());

    REQUIRE(loaded.size() == 1);

    auto& result = loaded.at("matrix");

    require_same_tensor(original, result);

    REQUIRE(result.at<float>({0, 0}) == 1.0f);
    REQUIRE(result.at<float>({0, 1}) == 2.0f);
    REQUIRE(result.at<float>({0, 2}) == 3.0f);
    REQUIRE(result.at<float>({1, 0}) == -4.0f);
    REQUIRE(result.at<float>({1, 1}) == -5.0f);
    REQUIRE(result.at<float>({1, 2}) == -6.0f);
}

TEST_CASE("Safetensors: save and load a 3D tensor")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_3d.safetensors");

    auto original = make_3d_tensor();

    yann::utils::save_safetensors(
        {{"tensor", original}},
        file.path.string()
    );

    auto loaded = yann::utils::load_safetensors(file.path.string());

    auto& result = loaded.at("tensor");

    require_same_tensor(original, result);
}

TEST_CASE("Safetensors: save and load multiple tensors")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_multiple.safetensors");

    auto vector = make_vector();
    auto matrix = make_matrix();
    auto tensor_3d = make_3d_tensor();

    yann::utils::save_safetensors(
        {
            {"vector", vector},
            {"matrix", matrix},
            {"tensor_3d", tensor_3d},
        },
        file.path.string()
    );

    auto loaded = yann::utils::load_safetensors(file.path.string());

    REQUIRE(loaded.size() == 3);
    REQUIRE(loaded.contains("vector"));
    REQUIRE(loaded.contains("matrix"));
    REQUIRE(loaded.contains("tensor_3d"));

    auto& loaded_vector = loaded.at("vector");
    auto& loaded_matrix = loaded.at("matrix");
    auto& loaded_3d = loaded.at("tensor_3d");

    require_same_tensor(vector, loaded_vector);
    require_same_tensor(matrix, loaded_matrix);
    require_same_tensor(tensor_3d, loaded_3d);
}

TEST_CASE("Safetensors: metadata does not prevent loading")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_metadata.safetensors");

    auto original = make_vector();

    yann::utils::save_safetensors(
        {{"tensor", original}},
        file.path.string(),
        {
            {"author", "yann"},
            {"description", "test tensor"},
        }
    );

    auto loaded = yann::utils::load_safetensors(file.path.string());

    REQUIRE(loaded.size() == 1);

    auto& result = loaded.at("tensor");

    require_same_tensor(original, result);
}

TEST_CASE("Safetensors: empty tensor map can be saved")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_empty.safetensors");

    yann::utils::save_safetensors(
        {},
        file.path.string()
    );

    auto loaded = yann::utils::load_safetensors(file.path.string());

    REQUIRE(loaded.empty());
}

TEST_CASE("Safetensors: loading a nonexistent file throws")
{
    cum::cum(cum::DEVICE::CPU);

    REQUIRE_THROWS(
        yann::utils::load_safetensors(
            "/this/file/definitely/does/not/exist.safetensors"
        )
    );
}

TEST_CASE("Safetensors: truncated file throws")
{
    cum::cum(cum::DEVICE::CPU);

    TempFile file("yann_safetensors_truncated.safetensors");

    {
        std::ofstream out(file.path, std::ios::binary);
        REQUIRE(out.good());

        const std::uint64_t header_size = 100;
        out.write(
            reinterpret_cast<const char*>(&header_size),
            sizeof(header_size)
        );
    }

    REQUIRE_THROWS(
        yann::utils::load_safetensors(file.path.string())
    );
}

TEST_CASE("Safetensors: debug clone failures", "[safetensors][debug]")
{
    cum::cum(cum::DEVICE::CPU);

    SECTION("clone vector")
    {
        cum::Tensor tensor(cum::Shape{6}, cum::datatype::FP32);

        for(int i = 0; i < 6; ++i)
            tensor.at<float>({i}) = static_cast<float>(i + 1);

        std::println(
            "CLONING vector: rank={} type={} format={} size={}",
            tensor.rank(),
            static_cast<int>(tensor.type()),
            static_cast<int>(tensor.format()),
            tensor.size()
        );

        auto clone = tensor.clone();

        std::println("CLONED vector");
        REQUIRE(clone.shape() == tensor.shape());
    }

    SECTION("clone matrix")
    {
        cum::Tensor tensor(cum::Shape{2, 3}, cum::datatype::FP32);

        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 3; ++j)
                tensor.at<float>({i, j}) =
                    static_cast<float>(i * 3 + j + 1);

        std::println(
            "CLONING matrix: rank={} type={} format={} size={}",
            tensor.rank(),
            static_cast<int>(tensor.type()),
            static_cast<int>(tensor.format()),
            tensor.size()
        );

        auto clone = tensor.clone();

        std::println("CLONED matrix");
        REQUIRE(clone.shape() == tensor.shape());
    }

    SECTION("clone 3D")
    {
        cum::Tensor tensor(cum::Shape{2, 2, 2}, cum::datatype::FP32);

        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 2; ++k)
                    tensor.at<float>({i, j, k}) =
                        static_cast<float>(i * 4 + j * 2 + k + 1);

        std::println(
            "CLONING 3D: rank={} type={} format={} size={}",
            tensor.rank(),
            static_cast<int>(tensor.type()),
            static_cast<int>(tensor.format()),
            tensor.size()
        );

        auto clone = tensor.clone();

        std::println("CLONED 3D");
        REQUIRE(clone.shape() == tensor.shape());
    }

    SECTION("clone tensor with explicit AB layout")
    {
        cum::Tensor tensor(
            cum::Shape{2, 3},
            cum::datatype::FP32,
            cum::layout::AB
        );

        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 3; ++j)
                tensor.at<float>({i, j}) =
                    static_cast<float>(i * 3 + j + 1);

        std::println(
            "CLONING explicit AB: rank={} type={} format={} size={}",
            tensor.rank(),
            static_cast<int>(tensor.type()),
            static_cast<int>(tensor.format()),
            tensor.size()
        );

        auto clone = tensor.clone();

        std::println("CLONED explicit AB");
        REQUIRE(clone.shape() == tensor.shape());
    }
}