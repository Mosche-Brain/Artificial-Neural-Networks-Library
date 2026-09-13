# CUM - **C**++ N**um**eric Library
Vendor agnostic C++ library providing linear algebra, eltwise operations, convenient datastructures and deep learning primitives without template hell (we moved the problems from the compile time to the runtime)

### CUM modules
* `cum/LinearAlgebra` - Classic BLAS routines and some convenience wrappers.
* `cum/functions` - Elementwise functions and some general stuff.
* `cum/neural_primitives` - Set of primitives for deep learning and N-Dimensional tensors operation.
* `cum/` - Memory and runtime management and `Vector`/`Matrix`/`Tensor` class

### Current development goal
* Working n-dimensional tensors
* Multi-device queue

### Known issues
* FP8 and lower floating point precision tensors didn't work and wouldn't for some time.
* Tensor class operator overloads and member functions aren't yet finished.

### Related projects ( most are not even existing )
* cumDNN *(also known as yann)*
* cumBLAS
* ~~cumLAPACK~~
* ~~cumFFT~~
* ~~cumGRAD~~
* ~~cumPy~~
* ~~cumGL~~
* ~~cumGPT~~
* ~~cumAGI~~
* ~~cumMileniumRun~~
* Cumming Simulator 2023