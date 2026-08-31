# YANN Math API guide
## 👀 Overview
YANN Math API is simple abstraction layer created for convenient switching math backend and numeric precisions. It provides basic linear algebra objects and opeations essential for neural networks.
## ⚙️ Configuring building options
To configure build options you must add defines in building command or edit `build_config.hpp` before running building proces.
### Selection of numeric precision
```cpp
#define USE_X_PRICISION
```
| Insted `x` you must put them one of the following:
* `DOUBLE` (float64)
* `SINGLE` (float32)
* `HALF` (float16)
* `BRAIN` (bfloat16)
* `QUARTER` (int8)

If you didn't define any precision library will use float32. \
Please notice that not every hardware gives you performance boost if your choose lower numeric precision. For example many older CPU's didn't have optimimalizations for operating on low precision floating points numbers and processors treat them as float32.

### Selecting computational API
```cpp
#define USE_X
```
| Instead x you must put here one of the following:
* `EIGEN3` - Currently only supported computational backend. Eigen3 allow us to perform parallel computations on CPU and speed up inference to even more than 8x in comparision to single core computations.
* `ONEAPI` - Planned for implementation after full finishing of `EIGEN3`. OneAPI is unified API for performing parallel computations on both CPU's and GPU's provided by Intel but also work on chips of other companies.
* `CUDA` - Maybe I will also implement CUDA support in future. CUDA is closed hardware-specyfic API provided by NV*DIA and the most commonly used API for performing ML tasks.


## ➗ Types and operations;
All matrices functions and typed are in `YANN::math_api`, to avoid writing too many characters in each line you can just place this line:
```cpp
using namespace YANN::math_api;
```


### Defining 
Creating matrix with demandem numbers of rows and columns. \
`YANN::matrix_t YANN::math_api::createMatrix(size_t rows, size_t cols);` 

Creating matrix filled by desired value. \
`YANN::matrix_t YANN::math_api::createMatrix(size_t rows, size_t cols, YANN::numeric_t value)` 

Creating matrix filled with random numbers \
`YANN::matrix_t YANN::math_api::createRandomMatrix(size_t rows, size_t cols, YANN::numeric_t value )` 

