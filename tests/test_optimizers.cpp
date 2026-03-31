#include <gtest/gtest.h>
#include <lunaml/lunaml.hpp>
using namespace nf;

TEST(OptimizerTest, SGDStep) {
    auto w = Tensor::full({2, 2}, 1.0f, true);
    auto x = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    auto y = (w * x).sum();
    y.backward();

    auto params = std::vector<Tensor*>{&w};
    optim::SGD sgd(params, 0.1f);
    sgd.step();

    // w should have decreased by lr * grad
    EXPECT_LT(w.data()[0], 1.0f);
}

TEST(OptimizerTest, AdamStep) {
    Random::manual_seed(42);
    auto w = Tensor::full({2}, 5.0f, true);
    auto target = Tensor::from_data({1.0f, 1.0f}, {2});

    auto params = std::vector<Tensor*>{&w};
    optim::Adam adam(params, 0.1f);

    for (int i = 0; i < 100; ++i) {
        auto diff = w - target;
        auto loss = (diff * diff).sum();
        adam.zero_grad();
        loss.backward();
        adam.step();
    }
    // Should be close to target
    EXPECT_NEAR(w.data()[0], 1.0f, 0.5f);
    EXPECT_NEAR(w.data()[1], 1.0f, 0.5f);
}

TEST(OptimizerTest, ZeroGrad) {
    auto w = Tensor::ones({2}, true);
    auto loss = w.sum();
    loss.backward();
    EXPECT_FLOAT_EQ(w.grad().data()[0], 1.0f);

    auto params = std::vector<Tensor*>{&w};
    optim::SGD sgd(params, 0.1f);
    sgd.zero_grad();
    EXPECT_FLOAT_EQ(w.grad().data()[0], 0.0f);
}
