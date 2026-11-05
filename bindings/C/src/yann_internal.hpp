#pragma once

#include "yann.h"

#include <cum/Tensor.hpp>
#include <cum/cum.hpp>

#include "yann/Parameter.hpp"
#include "yann/loss/LossBase.hpp"
#include "yann/models/Chain.hpp"
#include "yann/models/Sequential.hpp"
#include "yann/optimizers/OptimizerBase.hpp"

#include <memory>
#include <string>

struct yann_context_s
{
    cum::DEVICE device = cum::DEVICE::AUTO;
};

struct yann_tensor_s
{
    cum::Tensor tensor;
};

struct yann_parameter_s
{
    yann::Parameter param;
};

struct yann_model_s
{
    yann::models::Sequential model;
};

struct yann_chain_s
{
    yann::models::Chain chain;
};

struct yann_optimizer_s
{
    std::unique_ptr<yann::optimizers::OptimizerBase> optimizer;
};

struct yann_loss_s
{
    std::unique_ptr<yann::loss::LossBase> loss;
};

inline thread_local std::string g_last_error;

inline void yann_set_error(const std::string& msg)
{
    g_last_error = msg;
}

inline void yann_set_error(const char* msg)
{
    g_last_error = msg ? msg : "Unknown error";
}
