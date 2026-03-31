#pragma once
#include "lunaml/nn/module.hpp"
#include <vector>
#include <memory>

namespace nf { namespace nn {

class Sequential : public Module {
public:
    Sequential() = default;

    template<typename T, typename... Args>
    Sequential& add(Args&&... args) {
        layers_.push_back(std::make_shared<T>(std::forward<Args>(args)...));
        return *this;
    }

    Sequential& add(std::shared_ptr<Module> layer) {
        layers_.push_back(layer);
        return *this;
    }

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Sequential"; }

    size_t size() const { return layers_.size(); }
    Module& operator[](size_t idx) { return *layers_[idx]; }

    void train();
    void eval();

private:
    std::vector<std::shared_ptr<Module>> layers_;
};

}} // namespace nf::nn
