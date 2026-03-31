#include <gtest/gtest.h>
#include <lunaml/lunaml.hpp>
using namespace nf;

TEST(LossTest, MSE) {
    auto pred = Tensor::from_data({1, 2, 3}, {3, 1});
    auto target = Tensor::from_data({1, 2, 3}, {3, 1});
    loss::MSE mse;
    auto l = mse(pred, target);
    EXPECT_NEAR(l.item(), 0.0f, 1e-6);
}

TEST(LossTest, MSENonZero) {
    auto pred = Tensor::from_data({2.0f}, {1, 1}, true);
    auto target = Tensor::from_data({0.0f}, {1, 1});
    loss::MSE mse;
    auto l = mse(pred, target);
    EXPECT_NEAR(l.item(), 4.0f, 1e-5);
}

TEST(LossTest, BCE) {
    auto pred = Tensor::from_data({0.5f}, {1, 1}, true);
    auto target = Tensor::from_data({1.0f}, {1, 1});
    loss::BCE bce;
    auto l = bce(pred, target);
    // -log(0.5) ≈ 0.6931
    EXPECT_NEAR(l.item(), 0.6931f, 0.01f);
}

TEST(LossTest, MSEBackward) {
    auto pred = Tensor::from_data({3.0f}, {1, 1}, true);
    auto target = Tensor::from_data({1.0f}, {1, 1});
    loss::MSE mse;
    auto l = mse(pred, target);
    l.backward();
    // d/dpred MSE = 2*(pred-target)/n = 2*(3-1)/1 = 4
    EXPECT_TRUE(pred.has_grad());
}
