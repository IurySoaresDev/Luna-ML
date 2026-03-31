#pragma once

// ══════════════════════════════════════════════
// LunaML — C++ Machine Learning Library
// ══════════════════════════════════════════════

// Core
#include "lunaml/core/types.hpp"
#include "lunaml/core/tensor.hpp"
#include "lunaml/core/tensor_ops.hpp"
#include "lunaml/core/autograd.hpp"
#include "lunaml/core/random.hpp"

// Neural Network Layers
#include "lunaml/nn/module.hpp"
#include "lunaml/nn/linear.hpp"
#include "lunaml/nn/activation.hpp"
#include "lunaml/nn/sequential.hpp"
#include "lunaml/nn/conv2d.hpp"
#include "lunaml/nn/pooling.hpp"
#include "lunaml/nn/batchnorm.hpp"
#include "lunaml/nn/dropout.hpp"
#include "lunaml/nn/flatten.hpp"

// Loss Functions
#include "lunaml/loss/loss.hpp"
#include "lunaml/loss/mse.hpp"
#include "lunaml/loss/cross_entropy.hpp"
#include "lunaml/loss/bce.hpp"

// Optimizers
#include "lunaml/optim/optimizer.hpp"
#include "lunaml/optim/sgd.hpp"
#include "lunaml/optim/adam.hpp"
#include "lunaml/optim/rmsprop.hpp"

// Data
#include "lunaml/data/dataset.hpp"
#include "lunaml/data/dataloader.hpp"
#include "lunaml/data/csv_dataset.hpp"

// Metrics
#include "lunaml/metrics/metrics.hpp"

// Utilities
#include "lunaml/utils/serialization.hpp"
#include "lunaml/utils/logger.hpp"
