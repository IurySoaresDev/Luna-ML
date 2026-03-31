#include "lunaml/data/csv_dataset.hpp"
#include <sstream>

namespace nf { namespace data {

CSVDataset::CSVDataset(const std::string& filepath, size_t target_col, bool has_header, char delimiter) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Cannot open CSV file: " + filepath);

    std::string line;
    if (has_header && std::getline(file, line)) {} // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, delimiter)) {
            row.push_back(std::stof(cell));
        }
        if (row.empty()) continue;
        if (num_features_ == 0) num_features_ = row.size() - 1;

        float target = row[target_col];
        row.erase(row.begin() + target_col);
        inputs_.push_back(row);
        targets_.push_back(target);
    }
}

std::pair<Tensor, Tensor> CSVDataset::get(size_t index) const {
    auto input = Tensor::from_data(inputs_[index], {num_features_});
    auto target = Tensor::from_data({targets_[index]}, {1});
    return {input, target};
}

}} // namespace nf::data
