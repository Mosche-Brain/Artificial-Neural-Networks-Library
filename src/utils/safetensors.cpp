/*
 * @author: jaro
 * @name:   safetensors
 * @file:   src/utils/safetensors.cpp
 * @date:   16 September 2026 21:02:45
 */

#define SAFETENSORS_MAX_DIM 8
#define SAFETENSORS_MAX_TENSORS 2048
#define SAFETENSORS_MAX_FILE_SIZE (2ULL << 40) // 2 TiB
#define SAFETENSORS_MAX_STRING_SIZE 2048
#define SAFETENSORS_MAX_METADATA_SIZE 8192

#include <algorithm>
#include <fstream>
#include <cstring>
#include <climits>
#include <string>

#include <fcntl.h> // What is this?
#include <unistd.h> // We must eliminate this dependency
#include <sys/mman.h> // This also

#include "yann/utils/safetensors.hpp"

// tbh, original version of this code is even stranger than mine

namespace yann::utils
{

    inline cum::datatype get_cum_dtype(const std::string &dtype_str)
    {
        static const std::unordered_map<std::string, cum::datatype> dtype_map = {
            // {"BOOL", torch::kBool},
            {"U8", cum::datatype::U8},
            {"I8", cum::datatype::S8},
            {"U16", cum::datatype::U16},
            {"I16", cum::datatype::S16},
            {"U32", cum::datatype::U32},
            {"I32", cum::datatype::S32},
            {"U64", cum::datatype::U64},
            {"I64", cum::datatype::S64},
            {"F16", cum::datatype::FP16},
            {"BF16", cum::datatype::BF16},
            {"F32", cum::datatype::FP32},
            {"F64", cum::datatype::FP64}};

        auto it = dtype_map.find(dtype_str);
        if (it != dtype_map.end())
        {
            return it->second;
        }

        throw std::runtime_error("Unknown dtype: " + dtype_str);
    }


    inline std::string get_safetensors_dtype(cum::datatype dtype)
    {
        static const std::unordered_map<cum::datatype, std::string> dtype_map = {
            // {torch::kBool, "BOOL"},
            {cum::datatype::U8, "U8"},
            {cum::datatype::S8, "I8"},
            {cum::datatype::U16, "U16"},
            {cum::datatype::S16, "I16"},
            {cum::datatype::U32, "U32"},
            {cum::datatype::S32, "I32"},
            {cum::datatype::U64, "U64"},
            {cum::datatype::S64, "I64"},
            {cum::datatype::FP16, "F16"},
            {cum::datatype::BF16, "BF16"},
            {cum::datatype::FP32, "F32"},
            {cum::datatype::FP64, "F64"}};

        auto it = dtype_map.find(dtype);
        if (it != dtype_map.end())
        {
            return it->second;
        }
        throw std::runtime_error("Unsupported dtype");
    }


    // Class definitions
    class SimpleJSONParser
    {
    private:
        const char *json;
        size_t pos;

        inline void skipWhitespace()
        {
            while (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')
                pos++;
        }

        inline std::string parseString()
        {
            std::string result;
            pos++; // Skip opening quote
            while (json[pos] != '"')
            {
                if (json[pos] == '\\')
                {
                    pos++;
                    if (json[pos] == 'u')
                    {
                        // Handle Unicode escape (simplified)
                        pos += 4;
                    }
                }
                result += json[pos++];
            }
            pos++; // Skip closing quote
            return result;
        }

        inline std::vector<int64_t> parseArray()
        {
            std::vector<int64_t> result;
            pos++; // Skip opening bracket
            while (json[pos] != ']')
            {
                skipWhitespace();
                size_t num_start = pos;
                while (std::isdigit(json[pos]))
                    pos++;
                result.push_back(std::stoll(std::string(json + num_start, pos - num_start)));
                skipWhitespace();
                if (json[pos] == ',')
                    pos++;
            }
            pos++; // Skip closing bracket
            return result;
        }

        inline std::array<size_t, 2> parseDataOffsets()
        {
            std::array<size_t, 2> result;
            pos++; // Skip opening bracket
            skipWhitespace();
            size_t num_start = pos;
            while (std::isdigit(json[pos]))
                pos++;
            result[0] = std::stoull(std::string(json + num_start, pos - num_start));
            skipWhitespace();
            pos++; // Skip comma
            skipWhitespace();
            num_start = pos;
            while (std::isdigit(json[pos]))
                pos++;
            result[1] = std::stoull(std::string(json + num_start, pos - num_start));
            skipWhitespace();
            pos++; // Skip closing bracket
            return result;
        }

        inline TensorInfo parseTensorInfo()
        {
            TensorInfo info;
            pos++; // Skip opening brace
            while (json[pos] != '}')
            {
                skipWhitespace();
                std::string key = parseString();
                skipWhitespace();
                pos++; // Skip colon
                skipWhitespace();
                if (key == "dtype")
                {
                    cum::datatype dtype;

                    std::string type_tag = parseString();

                    type_tag == "F64" ? dtype = cum::datatype::FP64 :
                    type_tag == "F32" ? dtype = cum::datatype::FP32 :
                    type_tag == "F16" ? dtype = cum::datatype::FP16 :
                    type_tag == "BF16" ? dtype = cum::datatype::BF16 :

                    info.dtype = dtype;
                }
                else if (key == "shape")
                {
                    info.shape = parseArray();
                }
                else if (key == "data_offsets")
                {
                    info.data_offsets = parseDataOffsets();
                }
                else
                {
                    // Skip unknown fields
                    while (json[pos] != ',' && json[pos] != '}')
                        pos++;
                }
                skipWhitespace();
                if (json[pos] == ',')
                    pos++;
            }
            pos++; // Skip closing brace
            return info;
        }

    public:
        inline SimpleJSONParser(const char *json_str) : json(json_str), pos(0) {}

        inline std::unordered_map<std::string, TensorInfo> parse()
        {
            std::unordered_map<std::string, TensorInfo> result;
            skipWhitespace();
            if (json[pos++] != '{')
                throw std::runtime_error("Expected object");
            while (json[pos] != '}')
            {
                skipWhitespace();
                std::string key = parseString();
                skipWhitespace();
                pos++; // Skip colon
                skipWhitespace();
                if (key != "__metadata__")
                {
                    result[key] = parseTensorInfo();
                }
                else
                {
                    // Skip metadata
                    while (json[pos] != ',' && json[pos] != '}')
                        pos++;
                }
                skipWhitespace();
                if (json[pos] == ',')
                    pos++;
            }
            return result;
        }
    };


    inline std::unordered_map<std::string, TensorInfo> parse_safetensors_header_info(const char *data, size_t size)
    {
        if (size < 8)
            throw std::runtime_error("Invalid file size");

        uint64_t header_size;
        std::memcpy(&header_size, data, sizeof(uint64_t));

        if (8 + header_size > size)
            throw std::runtime_error("Invalid header size");

        SimpleJSONParser parser(data + 8);
        return parser.parse();
    }

    inline void validate_string_length(const std::string &str, const std::string &context)
    {
        if (str.length() > SAFETENSORS_MAX_STRING_SIZE)
        {
            throw std::runtime_error(context + " exceeds maximum allowed length");
        }
    }

    inline bool is_big_endian()
    {
        union
        {
            uint32_t i;
            char c[4];
        } bint = {0x01020304};

        return bint.c[0] == 1;
    }

    template <typename T>
    inline T swap_endian(T u)
    {
        static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

        union
        {
            T u;
            unsigned char u8[sizeof(T)];
        } source, dest;

        source.u = u;

        for (size_t k = 0; k < sizeof(T); k++)
            dest.u8[k] = source.u8[sizeof(T) - k - 1];

        return dest.u;
    }

    inline std::unordered_map<std::string, cum::Tensor> load_safetensors(const std::string &filename)
    {
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1)
        {
            close(fd);
            throw std::runtime_error("Failed to get file size");
        }
        size_t file_size = sb.st_size;

        if (file_size > SAFETENSORS_MAX_FILE_SIZE)
        {
            close(fd);
            throw std::runtime_error("File size exceeds maximum allowed size");
        }

        void *mapped_file = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_file == MAP_FAILED)
        {
            close(fd);
            throw std::runtime_error("Failed to memory map file");
        }

        try
        {
            uint64_t header_size;
            std::memcpy(&header_size, mapped_file, sizeof(uint64_t));
            if (is_big_endian())
            {
                header_size = swap_endian(header_size);
            }

            if (8 + header_size > file_size)
                throw std::runtime_error("Invalid header size");

            auto tensor_infos = parse_safetensors_header_info(static_cast<char *>(mapped_file), file_size);

            if (tensor_infos.size() > SAFETENSORS_MAX_TENSORS)
            {
                throw std::runtime_error("Number of tensors exceeds maximum allowed");
            }

            std::unordered_map<std::string, cum::Tensor> tensors;
            char *data_start = static_cast<char *>(mapped_file) + 8 + header_size;

            for (const auto &[name, info] : tensor_infos)
            {
                validate_string_length(name, "Tensor name");

                if (info.shape.size() > SAFETENSORS_MAX_DIM)
                {
                    throw std::runtime_error("Tensor dimension exceeds maximum allowed");
                }

                // cum::datatype dtype = info.dtype);

                // auto options = cum::TensorOptions()
                //                    .dtype(dtype)
                //                    .device(torch::kCPU);

                // cum::Tensor cpu_tensor = torch::from_blob(
                //                                data_start + info.data_offsets[0],
                //                                info.shape,
                //                                options)
                //                                .clone(); // Clone to own the data

                cum::Tensor tensor(info.shape, info.dtype);

                if (is_big_endian() && (info.dtype == cum::datatype::FP16 || info.dtype == cum::datatype::FP32 || info.dtype == cum::datatype::FP64))
                {
                    // auto data_ptr = static_cast<char *>(cpu_tensor.data_ptr());
                    auto data_ptr = tensor.data<char>();
                    cum::dim_t datatype_size = cum::datatype_size(tensor.type());
                    for (cum::dim_t i = 0; i < tensor.size(); i += datatype_size)
                    {
                        std::reverse(data_ptr + i, data_ptr + i + datatype_size);
                    }
                }

                tensors[name] = tensor;
            }

            munmap(mapped_file, file_size);
            close(fd);

            return tensors;
        }
        catch (...)
        {
            munmap(mapped_file, file_size);
            close(fd);
            throw;
        }
    }

    inline void save_safetensors(const std::unordered_map<std::string, cum::Tensor> &tensors, const std::string &filename, const std::unordered_map<std::string, std::string> &metadata = {})
    {
        if (tensors.size() > SAFETENSORS_MAX_TENSORS)
        {
            throw std::runtime_error("Number of tensors exceeds maximum allowed");
        }

        std::string header_json = "{";
        std::vector<char> data_buffer;
        size_t current_offset = 0;

        if (!metadata.empty())
        {
            header_json += "\"__metadata__\":{";
            bool first_meta = true;
            for (const auto &[key, value] : metadata)
            {
                validate_string_length(key, "Metadata key");
                validate_string_length(value, "Metadata value");

                if (!first_meta)
                    header_json += ",";
                header_json += "\"" + key + "\":\"" + value + "\"";
                first_meta = false;
            }
            header_json += "},";
        }

        for (const auto &[name, tensor] : tensors)
        {
            validate_string_length(name, "Tensor name");


            if (tensor.type() == cum::datatype::FP16 || tensor.type() == cum::datatype::FP32 || tensor.type() == cum::datatype::FP64)
            {
                // tensor = tensor.to(torch::kCPU, tensor.dtype(), /*non_blocking=*/false, /*copy=*/true);
                // auto data_ptr = static_cast<char *>(tensor.data_ptr());
                auto data_ptr = tensor.data<char>();
                cum::dim_t datatype_size = cum::datatype_size(tensor.type());
                for (cum::dim_t i = 0; i < tensor.size() ; i += datatype_size)
                {
                    std::reverse(data_ptr + i, data_ptr + i + datatype_size);
                }
            }

            if (tensor.rank() > SAFETENSORS_MAX_DIM)
            {
                throw std::runtime_error("Tensor dimension exceeds maximum allowed");
            }

            auto dtype = get_safetensors_dtype(tensor.type());
            // auto shape = tensor.sizes().vec();
            auto shape = tensor.shape();
            // size_t tensor_size = tensor.numel() * tensor.element_size();
            size_t tensor_size = tensor.size();

            if (header_json.length() > 1)
                header_json += ",";
            header_json += "\"" + name + "\":{";
            header_json += "\"dtype\":\"" + dtype + "\",";
            header_json += "\"shape\":[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0)
                    header_json += ",";
                header_json += std::to_string(shape[i]);
            }
            header_json += "],";
            header_json += "\"data_offsets\":[" + std::to_string(current_offset) + "," + std::to_string(current_offset + tensor_size) + "]";
            header_json += "}";

            // const char *tensor_data = static_cast<const char *>(tensor.data_ptr());
            const char* tensor_data = tensor.data<const char>();
            data_buffer.insert(data_buffer.end(), tensor_data, tensor_data + tensor_size);

            current_offset += tensor_size;
        }

        header_json += "}";
        uint64_t header_size = header_json.size();

        if (header_size > SAFETENSORS_MAX_METADATA_SIZE)
        {
            throw std::runtime_error("Metadata size exceeds maximum allowed size");
        }

        if (8 + header_size + data_buffer.size() > SAFETENSORS_MAX_FILE_SIZE)
        {
            throw std::runtime_error("Total file size exceeds maximum allowed size");
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        uint64_t little_endian_header_size = header_size;
        if (is_big_endian())
        {
            little_endian_header_size = swap_endian(header_size);
        }
        file.write(reinterpret_cast<const char *>(&little_endian_header_size), sizeof(uint64_t));

        file.write(header_json.data(), header_json.size());

        file.write(data_buffer.data(), data_buffer.size());

        if (!file)
        {
            throw std::runtime_error("Failed to write to file: " + filename);
        }
    }

}