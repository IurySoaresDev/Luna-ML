#include "lunaml/data/dataloader.hpp"

namespace nf { namespace data {

std::pair<Tensor, Tensor> DataLoader::Iterator::operator*() {
    size_t end = std::min(pos_ + dl_.batch_size_, dl_.dataset_.size());
    size_t actual_batch = end - pos_;

    // Get first sample to determine shapes
    auto [first_input, first_target] = dl_.dataset_.get(dl_.indices_[pos_]);
    Shape input_shape = first_input.shape();
    Shape target_shape = first_target.shape();

    // Build batch shapes
    Shape batch_input_shape = {actual_batch};
    for (auto d : input_shape) batch_input_shape.push_back(d);
    Shape batch_target_shape = {actual_batch};
    for (auto d : target_shape) batch_target_shape.push_back(d);

    Tensor batch_input(batch_input_shape, false);
    Tensor batch_target(batch_target_shape, false);

    size_t input_stride = first_input.size();
    size_t target_stride = first_target.size();

    // Copy first sample
    std::copy(first_input.data(), first_input.data() + input_stride, batch_input.data());
    std::copy(first_target.data(), first_target.data() + target_stride, batch_target.data());

    // Copy remaining samples
    for (size_t i = 1; i < actual_batch; ++i) {
        auto [inp, tgt] = dl_.dataset_.get(dl_.indices_[pos_ + i]);
        std::copy(inp.data(), inp.data() + input_stride, batch_input.data() + i * input_stride);
        std::copy(tgt.data(), tgt.data() + target_stride, batch_target.data() + i * target_stride);
    }

    return {batch_input, batch_target};
}

}} // namespace nf::data
