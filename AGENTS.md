`/AGENTS.md`

```markdown
# AGENTS.md

## Project Overview

YANN is a C++23 High Performance Computing and Machine Learning framework.

The project is split into two major layers:

1. `cum` - low-level computational backend
2. `yann` - high-level machine learning framework built on top of `cum`

The main goals of the project are:

- high performance numerical computing
- machine learning workloads
- portable acceleration through SYCL
- support for multiple execution backends
- clean C++23 API design

---

# Architecture

## Dependency direction

The dependency graph must always remain:

```

yann
|
v
cum
|
v
backend implementations

```

Rules:

- `cum` MUST NOT depend on `yann`.
- Low-level numerical code MUST NOT know about high-level ML concepts.
- Backend-specific implementations MUST stay inside backend modules.
- Avoid introducing circular dependencies.

---

# cum module

Location:

```

modules/cum

```

`cum` is the numerical computing layer.

It provides:

- Matrix
- Vector
- Tensor
- MatrixView
- Linear algebra operations
- BLAS operations
- Neural primitives
- Mathematical functions
- Memory management
- Runtime abstraction

Important headers:

```

cum/Matrix.hpp
cum/Vector.hpp
cum/Tensor.hpp
cum/MatrixView.hpp
cum/LinearAlgebra.hpp
cum/memory.hpp
cum/runtime.hpp

````

Prefer existing `cum` abstractions instead of introducing new numerical types.

DO NOT replace:

```cpp
cum::Matrix
cum::Vector
cum::Tensor
cum::MatrixView
````

with:

```cpp
std::vector
raw arrays
other external numerical libraries
```

unless explicitly requested.

---

# yann module

Location:

```
include/yann
src
```

`yann` provides the machine learning framework.

Main components:

```
models/
    layers/
    ModelBase
    Sequential
    Perceptron
    NGram

optimizers/
    OptimizerBase
    SGD
    ADAM

loss/
    LossBase
    BinaryCrossEntropy
    MeanSquaredError
    MeanAbsoluteError
```

High-level code should use `cum` for numerical operations.

Do not implement custom matrix or tensor logic inside `yann`.

---

# C++ Rules

Target standard:

```
C++23
```

Prefer:

* RAII
* value semantics
* move semantics
* constexpr where useful
* noexcept where correct
* strong types
* clear ownership

Avoid:

* unnecessary heap allocations
* unnecessary shared ownership
* raw owning pointers
* hidden copies
* unnecessary abstractions

Generated code should compile with the existing project.

---

# Architecture Rules

## Runtime polymorphism

Runtime polymorphism is intentional in high-level components.

Examples:

```cpp
LayerBase
OptimizerBase
ModelBase
```

Virtual interfaces are acceptable here.

Do not remove polymorphism only to "optimize".

---

## Low-level data structures

Classes such as:

```cpp
cum::Tensor
cum::Matrix
cum::Vector
```

are performance-critical.

Avoid:

* virtual functions
* dynamic allocation in hot paths
* unnecessary indirection
* hidden copies

Prefer predictable memory layout and efficient access patterns.

---

# Memory and Performance Rules

Always consider:

* cache locality
* contiguous memory
* allocation frequency
* memory transfers
* synchronization points
* temporary objects
* data ownership

Avoid:

```cpp
allocate inside training loops
```

Prefer:

```cpp
reuse existing buffers
```

Avoid unnecessary:

```cpp
queue.wait()
```

Prefer asynchronous execution when possible.

Do not optimize blindly.

A small theoretical improvement is not worth reducing maintainability.

---

# SYCL and oneAPI Rules

Primary compute model:

```
SYCL 2020
```

Prefer standard SYCL APIs.

Intel oneAPI extensions are allowed when:

* they provide meaningful performance improvements,
* expose required hardware capabilities,
* are explicitly requested.

Do not make the entire project dependent on Intel-only functionality without a strong reason.

Keep backend portability in mind.

---

# Backend Rules

Backend implementations are located under:

```
modules/cum/sources/Backends
```

Current backends include:

```
MKL
Native
CBLAS
```

Backend-specific code belongs only inside backend directories.

Do not leak backend details into public APIs.

Example:

Bad:

```cpp
yann::Layer uses MKL specific types
```

Good:

```cpp
yann uses cum API
cum selects backend internally
```

---

# ML Framework Rules

When modifying layers:

Prefer existing abstractions:

```cpp
Parameter
ForwardCache
cum::Matrix
cum::Vector
```

Avoid creating duplicated storage.

Layers usually operate on:

```cpp
cum::Matrix
```

not raw arrays.

Reuse caches where possible.

Do not allocate new matrices unnecessarily during:

* forward pass
* backward pass
* optimization steps

---

# Testing Rules

When adding functionality:

* add focused tests
* keep tests deterministic
* test numerical correctness first
* consider backend differences

Prefer small reproducible examples.

Examples are located in:

```
examples/
```

Tests are located in:

```
tests/
```

---

# Code Modification Rules

When changing existing code:

* make the smallest reasonable change
* preserve existing API
* preserve naming conventions
* preserve formatting style
* avoid unrelated refactoring

Do not redesign architecture unless explicitly requested.

---

# Forbidden Changes

Do not:

* introduce Eigen
* introduce xtensor
* replace cum containers with STL containers
* add dependencies without justification
* move backend code into yann
* add yann dependencies into cum
* create duplicate numerical implementations
* invent APIs that do not exist

---

# Agent Behaviour

When unsure:

1. Inspect existing code.
2. Follow existing patterns.
3. Prefer consistency over personal preference.
4. Ask questions if multiple architectural choices are possible.

Do not invent missing APIs or classes.

If proposing optimization:

Explain:

* expected benefit
* trade-off
* affected component
