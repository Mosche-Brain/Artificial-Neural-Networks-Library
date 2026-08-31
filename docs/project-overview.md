# Project Overview: YANN

YANN is a C++23 High Performance Computing (HPC) and Machine Learning framework designed for high-performance numerical computing and machine learning workloads.

## Architecture Overview

The project is structured into two primary layers:

1.  **`cum` (Low-level Computational Backend):**
    *   **Location:** `modules/cum`
    *   **Responsibility:** Handles all low-level numerical operations.
    *   **Key Components:** Matrix, Vector, Tensor, MatrixView, Linear Algebra operations, BLAS, Neural primitives, Mathematical functions, Memory management, and Runtime abstraction.
    *   **Key Headers:** `cum/Matrix.hpp`, `cum/Vector.hpp`, `cum/Tensor.hpp`, `cum/MatrixView.hpp`, `cum/LinearAlgebra.hpp`, `cum/memory.hpp`, `cum/runtime.hpp`.
    *   **Design Principle:** Performance-critical. Avoids virtual functions, dynamic allocation in hot paths, and unnecessary indirection.

2.  **`yann` (High-level ML Framework):**
    *   **Location:** `include/yann`, `src`
    *   **Responsibility:** Provides the machine learning framework built on top of `cum`.
    *   **Key Components:**
        *   `models/`: Layers (ModelBase, Sequential, Perceptron, NGram), etc.
        *   `optimizers/`: OptimizerBase, SGD, ADAM, etc.
        *   `loss/`: LossBase, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError, etc.
    *   **Design Principle:** Uses `cum` for all numerical operations. Implements runtime polymorphism where appropriate (e.g., for Layers, Optimizers, and Models).

## Dependency Rules

The dependency graph must strictly follow this flow:
`yann` -> `cum` -> `backend implementations`

*   `cum` must **never** depend on `yann`.
*   Low-level numerical code must remain agnostic of high-level ML concepts.
*   Backend-specific implementations must stay within their respective backend modules.

## Development Guidelines

### C++ & Performance
*   **Standard:** C++23.
*   **Best Practices:** RAII, value/move semantics, `constexpr`, `noexcept`, and strong types.
*   **Memory Management:**
    *   Avoid heap allocations inside training loops.
    *   Reuse existing buffers where possible.
    *   Optimize for cache locality and contiguous memory.
*   **Execution:** Prefer asynchronous execution and avoid unnecessary `queue.wait()` calls.

### SYCL & Backends
*   **Compute Model:** SYCL 2020.
*   **Backends:** MKL, Native, and CBLAS are currently supported.
*   **Abstraction:** Backend-specific details must not leak into public APIs.

### ML Framework Specifics
*   Layers should operate on `cum::Matrix` (not raw arrays).
*   Avoid duplicated storage; reuse caches during forward and backward passes.
*   Do not allocate new matrices unnecessarily during training steps.

### Forbidden Actions
*   No introduction of Eigen or xtensor.
*   Do not replace `cum` containers with STL containers.
*   No moving backend code into `yann` or adding `yann` dependencies into `cum`.
*   No duplicate numerical implementations.
