#pragma once
#include "lunaml/data/dataset.hpp"
#include <string>
#include <fstream>

namespace nf { namespace data {

class CSVDataset : public Dataset {
public:
    CSVDataset(const std::string& filepath, size_t target_col, bool has_header = true, char delimiter = ',');
    size_t size() const override { return inputs_.size(); }
    std::pair<Tensor, Tensor> get(size_t index) const override;
    size_t num_features() const { return num_features_; }

private:
    std::vector<std::vector<float>> inputs_;
    std::vector<float> targets_;
    size_t num_features_ = 0;
};

}} // namespace nf::data
