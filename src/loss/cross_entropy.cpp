#include "lunaml/loss/cross_entropy.hpp"

namespace nf { namespace loss {

Tensor CrossEntropy::forward(const Tensor& prediction, const Tensor& target) {
    // prediction: [batch, classes] (logits)
    // target: [batch, 1] or [batch] (class indices)
    auto log_probs = nf::log_softmax(prediction, 1);  // [batch, classes]
    size_t batch = prediction.shape(0);
    size_t classes = prediction.shape(1);

    // NLL loss
    Tensor loss({1}, prediction.requires_grad());
    float total = 0;
    for (size_t b = 0; b < batch; ++b) {
        size_t cls = static_cast<size_t>(target.data()[b]);
        total -= log_probs.data()[b * classes + cls];
    }
    loss.data()[0] = total / static_cast<float>(batch);

    // For autograd, use the differentiable path
    // -sum(one_hot * log_softmax) / batch
    if (prediction.requires_grad()) {
        // Create one-hot
        Tensor one_hot = Tensor::zeros({batch, classes}, false);
        for (size_t b = 0; b < batch; ++b) {
            size_t cls = static_cast<size_t>(target.data()[b]);
            one_hot.data()[b * classes + cls] = 1.0f;
        }
        auto nll = -(one_hot * log_probs).sum() / static_cast<float>(batch);
        return nll;
    }
    return loss;
}

}} // namespace nf::loss
