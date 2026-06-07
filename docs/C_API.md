# YANN C API Documentation

This document explains how to use the **C API** of YANN (also referred to as `cYANN`).

The C API provides a stable, language-agnostic interface to YANN's neural network functionality. It is designed to be resilient to internal C++ changes and can be used from C, C++, Python (via `ctypes`), Rust, and other languages.

---

## 1. Overview

The C API is built as a separate library called **`cYANN`**.

- **Header**: `yann.h`
- **Library target**: `cYANN`
- **Design goals**:
  - Stable ABI
  - Opaque handles (implementation details are hidden)
  - Explicit resource management (`create` / `destroy`)
  - Error handling via return codes + `yann_get_last_error()`

**Important**: All data exchange at the C API boundary uses `float` (FP32), regardless of the internal precision (`cumeric_t`) the library was compiled with (FP16, FP32, FP64, etc.).

---

## 2. Building YANN with the C API

Enable the C API when configuring the project:

```bash
cmake .. \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCUM_USE_F16=ON \
  -DBUILD_USE_MKL=ON \
  -DBUILD_C_API=ON
```

Then build:

```bash
cmake --build . --target cYANN -j
```

After building, the library will be located at:

```
build/bindings/C/libcYANN.so     # or .a if built as static
```

The public header is located at:

```
bindings/C/include/yann.h
```

---

## 3. Basic Concepts

### Opaque Handles

All major objects are exposed through opaque pointers:

| Type                | Description                     | Must be freed with          |
|---------------------|---------------------------------|-----------------------------|
| `yann_context_t*`   | Library context / device state  | `yann_destroy_context()`    |
| `yann_model_t*`     | Neural network model            | `yann_model_destroy()`      |
| `yann_tensor_t*`    | Multi-dimensional data (matrix) | `yann_tensor_destroy()`     |

### Error Handling

Most functions return `yann_status_t`. On failure, you can get a human-readable message:

```c
if (status != YANN_OK) {
    printf("Error: %s\n", yann_get_last_error());
}
```

---

## 4. Basic Usage Example

Here is a complete example of creating a simple model and running inference:

```c
#include <yann/yann.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    yann_context_t* ctx = NULL;
    yann_model_t* model = NULL;
    yann_tensor_t* input = NULL;
    yann_tensor_t* output = NULL;

    // 1. Create context
    if (yann_create_context(&ctx) != YANN_OK) {
        fprintf(stderr, "Failed to create context: %s\n", yann_get_last_error());
        return 1;
    }

    // 2. Create model
    yann_model_create(ctx, &model);

    // 3. Add layers (Sequential style)
    yann_model_add_dense(model, 64, "relu");
    yann_model_add_dense(model, 32, "relu");
    yann_model_add_dense(model, 10, "linear");

    // 4. Prepare input data (example: batch of 1 sample with 784 features)
    float input_data[784] = {0}; // fill with real data...
    yann_tensor_create_from_host(ctx, 1, 784, input_data, &input);

    // 5. Run inference
    if (yann_model_predict(model, input, &output) != YANN_OK) {
        fprintf(stderr, "Inference failed: %s\n", yann_get_last_error());
        goto cleanup;
    }

    // 6. Get results
    float* result = malloc(10 * sizeof(float));
    yann_tensor_copy_to_host(output, result, 10);

    printf("Output: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", result[i]);
    }
    printf("\n");

    free(result);

cleanup:
    yann_tensor_destroy(output);
    yann_tensor_destroy(input);
    yann_model_destroy(model);
    yann_destroy_context(ctx);

    return 0;
}
```

---

## 5. API Reference (Core Functions)

### Context

```c
yann_status_t yann_create_context(yann_context_t** out_ctx);
void          yann_destroy_context(yann_context_t* ctx);
```

### Model

```c
yann_status_t yann_model_create(yann_context_t* ctx, yann_model_t** out_model);
void          yann_model_destroy(yann_model_t* model);

yann_status_t yann_model_add_dense(yann_model_t* model, size_t units, const char* activation);

yann_status_t yann_model_predict(yann_model_t* model,
                                 yann_tensor_t* input,
                                 yann_tensor_t** output);
```

**Supported activations** (as strings):
- `"relu"`
- `"linear"`
- `"sigmoid"`
- `"tanh"`
- `"softmax"`

### Tensor

```c
yann_status_t yann_tensor_create(yann_context_t* ctx, size_t rows, size_t cols, yann_tensor_t** out);
yann_status_t yann_tensor_create_from_host(yann_context_t* ctx, size_t rows, size_t cols,
                                           const float* data, yann_tensor_t** out);

yann_status_t yann_tensor_copy_to_host(yann_tensor_t* tensor, float* dst, size_t len);
yann_status_t yann_tensor_get_shape(yann_tensor_t* tensor, size_t* rows, size_t* cols);

void yann_tensor_destroy(yann_tensor_t* tensor);
```

---

## 6. Linking cYANN from Another Project

### Option A: Using Imported Target (Recommended for manual linking)

```cmake
add_library(cYANN IMPORTED SHARED)

set_target_properties(cYANN PROPERTIES
    IMPORTED_LOCATION "/path/to/libcYANN.so"
    INTERFACE_INCLUDE_DIRECTORIES "/path/to/Yann/bindings/C/include"
)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE cYANN)

# Required because cYANN uses SYCL
target_compile_options(my_app PRIVATE -fsycl)
target_link_options(my_app PRIVATE -fsycl)
```

### Option B: Using `add_subdirectory` (during development)

```cmake
add_subdirectory(${CMAKE_SOURCE_DIR}/../Yann ${CMAKE_BINARY_DIR}/yann)
target_link_libraries(my_app PRIVATE cYANN)
target_compile_options(my_app PRIVATE -fsycl)
target_link_options(my_app PRIVATE -fsycl)
```

### Option C: After Installation

```cmake
find_package(cYANN REQUIRED)
target_link_libraries(my_app PRIVATE cYANN)
```

---

## 7. Important Notes & Gotchas

### Compiler Requirements

- The library is built with Intel oneAPI (`icpx`).
- For best results, compile and link your application with `icpx` as well.
- You **must** pass `-fsycl` to both compile and link steps.

### Memory Management

- You are responsible for destroying every object you create.
- Tensors created inside `yann_model_predict` (the output) must be destroyed by the caller.

### Data Types

- The C API always uses `float` for input/output data.
- Internal computations use whatever precision YANN was compiled with (`CUM_USE_F16`, `CUM_USE_F32`, etc.).

### Thread Safety

Current status: **Not guaranteed**. Create separate contexts if using multiple threads.

---

## 8. Current Limitations (as of 2025)

- Only `Dense` layers are exposed via the C API.
- Training support (`yann_model_fit`) exists but is basic.
- No support for custom layers, optimizers configuration, or model saving/loading yet.
- Limited introspection (no easy way to query layer shapes after creation).

These limitations will be reduced as the C API matures.

---

## 9. Example Project Structure

```
MyApp/
├── CMakeLists.txt
├── main.c
└── build/
```

**CMakeLists.txt** (minimal):

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyApp)

add_library(cYANN IMPORTED SHARED)
set_target_properties(cYANN PROPERTIES
    IMPORTED_LOCATION "/home/user/Yann/build/bindings/C/libcYANN.so"
    INTERFACE_INCLUDE_DIRECTORIES "/home/user/Yann/bindings/C/include"
)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE cYANN)

target_compile_options(my_app PRIVATE -fsycl)
target_link_options(my_app PRIVATE -fsycl)
```

---

## 10. Getting Help

- Check `yann_get_last_error()` on every call during development.
- Enable debug output in YANN (if available) when building the main library.
- For advanced usage, inspect the header file: `bindings/C/include/yann.h`

---

**Happy hacking!**

The C API is still young. Feedback and contributions to make it more complete are welcome.