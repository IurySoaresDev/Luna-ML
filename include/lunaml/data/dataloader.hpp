#pragma once
#include "lunaml/data/dataset.hpp"
#include "lunaml/core/random.hpp"
#include <vector>
#include <algorithm>

namespace nf { namespace data {

class DataLoader {
public:
    DataLoader(Dataset& dataset, size_t batch_size, bool shuffle = true)
        : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {
        indices_.resize(dataset.size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    class Iterator {
    public:
        Iterator(DataLoader& dl, size_t pos) : dl_(dl), pos_(pos) {}
        std::pair<Tensor, Tensor> operator*();
        Iterator& operator++() { pos_ += dl_.batch_size_; return *this; }
        bool operator!=(const Iterator& other) const { return pos_ < other.pos_; }
    private:
        DataLoader& dl_;
        size_t pos_;
    };

    Iterator begin() {
        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), Random::generator());
        }
        return Iterator(*this, 0);
    }
    Iterator end() { return Iterator(*this, dataset_.size()); }
    size_t num_batches() const { return (dataset_.size() + batch_size_ - 1) / batch_size_; }

private:
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    friend class Iterator;
};

}} // namespace nf::data
