/**
 * @file YANN.hpp
 * @brief Main umbrella header for the YANN (Yet Another Neural Network) framework.
 *
 * Provides common classes and functions for constructing models, layers,
 * training routines, optimizers, and losses.
 */

#pragma once

#include "runtime_config.hpp"
#include "models/Sequential.hpp"
#include "models/layers/Layers.hpp"
#include "models/layers/LayerBase.hpp"
#include "models/layers/Dense.hpp"
#include "loss.hpp"
#include "logging.hpp"
