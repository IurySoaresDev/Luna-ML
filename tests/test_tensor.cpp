#include <gtest/gtest.h>
#include <lunaml/lunaml.hpp>
using namespace nf;

TEST(TensorTest, CreateZeros) {
    auto t = Tensor::zeros({3, 4});
    EXPECT_EQ(t.shape(0), 3);
    EXPECT_EQ(t.shape(1), 4);
    EXPECT_EQ(t.size(), 12);
    for (size_t i = 0; i < t.size(); ++i) EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
}

TEST(TensorTest, CreateOnes) {
    auto t = Tensor::ones({2, 3});
    EXPECT_EQ(t.size(), 6);
    for (size_t i = 0; i < t.size(); ++i) EXPECT_FLOAT_EQ(t.data()[i], 1.0f);
}

TEST(TensorTest, FromData) {
    auto t = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[5], 6.0f);
}

TEST(TensorTest, Addition) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::full({2, 3}, 2.0f);
    auto c = a + b;
    for (size_t i = 0; i < c.size(); ++i) EXPECT_FLOAT_EQ(c.data()[i], 3.0f);
}

TEST(TensorTest, Subtraction) {
    auto a = Tensor::full({2, 3}, 5.0f);
    auto b = Tensor::full({2, 3}, 3.0f);
    auto c = a - b;
    for (size_t i = 0; i < c.size(); ++i) EXPECT_FLOAT_EQ(c.data()[i], 2.0f);
}

TEST(TensorTest, Multiplication) {
    auto a = Tensor::full({2, 3}, 3.0f);
    auto b = Tensor::full({2, 3}, 4.0f);
    auto c = a * b;
    for (size_t i = 0; i < c.size(); ++i) EXPECT_FLOAT_EQ(c.data()[i], 12.0f);
}

TEST(TensorTest, MatMul) {
    auto a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto b = Tensor::from_data({7, 8, 9, 10, 11, 12}, {3, 2});
    auto c = a.matmul(b);
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 2);
    EXPECT_FLOAT_EQ(c.data()[0], 58.0f);   // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(c.data()[1], 64.0f);   // 1*8 + 2*10 + 3*12
}

TEST(TensorTest, Transpose) {
    auto a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto b = a.transpose();
    EXPECT_EQ(b.shape(0), 3);
    EXPECT_EQ(b.shape(1), 2);
    EXPECT_FLOAT_EQ(b.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(b.data()[1], 4.0f);
}

TEST(TensorTest, Sum) {
    auto a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto s = a.sum();
    EXPECT_FLOAT_EQ(s.item(), 21.0f);
}

TEST(TensorTest, Mean) {
    auto a = Tensor::from_data({2, 4, 6, 8}, {2, 2});
    auto m = a.mean();
    EXPECT_FLOAT_EQ(m.item(), 5.0f);
}

TEST(TensorTest, Reshape) {
    auto a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto b = a.reshape({3, 2});
    EXPECT_EQ(b.shape(0), 3);
    EXPECT_EQ(b.shape(1), 2);
}

TEST(TensorTest, ScalarOps) {
    auto a = Tensor::full({2, 2}, 3.0f);
    auto b = a * 2.0f;
    for (size_t i = 0; i < b.size(); ++i) EXPECT_FLOAT_EQ(b.data()[i], 6.0f);
    auto c = a + 1.0f;
    for (size_t i = 0; i < c.size(); ++i) EXPECT_FLOAT_EQ(c.data()[i], 4.0f);
}

TEST(TensorTest, Broadcasting) {
    auto a = Tensor::from_data({1, 2, 3}, {1, 3});
    auto b = Tensor::from_data({10, 20}, {2, 1});
    auto c = a + b;
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 3);
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[5], 23.0f);
}

TEST(TensorTest, Item) {
    auto a = Tensor::from_data({42.0f}, {1});
    EXPECT_FLOAT_EQ(a.item(), 42.0f);
}

TEST(TensorTest, Exp) {
    auto a = Tensor::from_data({0.0f, 1.0f}, {2});
    auto b = a.exp();
    EXPECT_NEAR(b.data()[0], 1.0f, 1e-5);
    EXPECT_NEAR(b.data()[1], std::exp(1.0f), 1e-5);
}
