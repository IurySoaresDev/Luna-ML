#include "lunaml/nn/sequential.hpp"

namespace nf { namespace nn {

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> all_params;
    for (auto& layer : layers_) {
        auto layer_params = layer->parameters();
        all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    return all_params;
}

void Sequential::train() {
    training_ = true;
    for (auto& layer : layers_) layer->train();
}

void Sequential::eval() {
    training_ = false;
    for (auto& layer : layers_) layer->eval();
}

}} // namespace nf::nn
