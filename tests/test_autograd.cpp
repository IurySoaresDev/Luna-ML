#include <gtest/gtest.h>
#include <lunaml/lunaml.hpp>
using namespace nf;

TEST(AutogradTest, SimpleAdd) {
    auto a = Tensor::from_data({2.0f, 3.0f}, {2}, true);
    auto b = Tensor::from_data({4.0f, 5.0f}, {2}, true);
    auto c = a + b;
    auto loss = c.sum();
    loss.backward();
    // d(a1+b1+a2+b2)/da_i = 1
    EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[1], 1.0f);
    EXPECT_FLOAT_EQ(b.grad().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(b.grad().data()[1], 1.0f);
}

TEST(AutogradTest, SimpleMul) {
    auto a = Tensor::from_data({2.0f}, {1}, true);
    auto b = Tensor::from_data({3.0f}, {1}, true);
    auto c = a * b;
    c.backward();
    EXPECT_FLOAT_EQ(a.grad().data()[0], 3.0f); // dc/da = b
    EXPECT_FLOAT_EQ(b.grad().data()[0], 2.0f); // dc/db = a
}

TEST(AutogradTest, MatMulBackward) {
    auto a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, true);
    auto b = Tensor::from_data({5, 6, 7, 8}, {2, 2}, true);
    auto c = a.matmul(b);
    auto loss = c.sum();
    loss.backward();
    // Gradients should be non-zero
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NE(a.grad().data()[i], 0.0f);
        EXPECT_NE(b.grad().data()[i], 0.0f);
    }
}

TEST(AutogradTest, ChainRule) {
    // f(x) = (x * 2 + 1)^2, x=3 => f'(x) = 2*(x*2+1)*2 = 4*(2*3+1)=28
    auto x = Tensor::from_data({3.0f}, {1}, true);
    auto y = x * 2.0f + 1.0f;
    auto z = y * y;
    z.backward();
    EXPECT_NEAR(x.grad().data()[0], 28.0f, 1e-4);
}

TEST(AutogradTest, NoGradGuard) {
    auto a = Tensor::from_data({2.0f}, {1}, true);
    Tensor c;
    {
        NoGradGuard guard;
        auto b = a * 3.0f;
        c = b + 1.0f;
    }
    EXPECT_EQ(c.grad_fn(), nullptr);
}

TEST(AutogradTest, ExpBackward) {
    auto x = Tensor::from_data({1.0f}, {1}, true);
    auto y = x.exp();
    y.backward();
    EXPECT_NEAR(x.grad().data()[0], std::exp(1.0f), 1e-5);
}

TEST(AutogradTest, NegBackward) {
    auto x = Tensor::from_data({3.0f}, {1}, true);
    auto y = -x;
    auto z = y.sum();
    z.backward();
    EXPECT_FLOAT_EQ(x.grad().data()[0], -1.0f);
}
