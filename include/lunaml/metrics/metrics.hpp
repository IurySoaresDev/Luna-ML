#pragma once
#include "lunaml/core/tensor.hpp"

namespace nf { namespace metrics {

float accuracy(const Tensor& predictions, const Tensor& targets);
float precision(const Tensor& predictions, const Tensor& targets, float threshold = 0.5f);
float recall(const Tensor& predictions, const Tensor& targets, float threshold = 0.5f);
float f1_score(const Tensor& predictions, const Tensor& targets, float threshold = 0.5f);

}} // namespace nf::metrics
