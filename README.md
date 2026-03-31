<p align="center">
  <h1 align="center">⚡ LunaML</h1>
  <p align="center">
    <strong>Biblioteca de Machine Learning em C++ moderno — do zero, sem dependências externas.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
    <img src="https://img.shields.io/badge/Build-CMake_3.16+-green.svg" alt="CMake">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT">
    <img src="https://img.shields.io/badge/GPU-CPU_Only_v1-lightgrey.svg" alt="CPU">
  </p>
</p>

---

## 📋 Visão Geral

**LunaML** é uma biblioteca de machine learning escrita do zero em C++17, projetada para ser educacional, modular e performática. Sem nenhuma dependência externa, ela implementa desde a álgebra linear fundamental até redes neurais completas com diferenciação automática.

### ✨ Features

- 🔢 **Tensor N-dimensional** — Armazenamento eficiente com operações element-wise, broadcasting e matmul
- 🔄 **Autograd** — Grafo computacional dinâmico (estilo PyTorch) com backpropagation automática
- 🧠 **Camadas Neurais** — Linear, Conv2D, MaxPool2D, BatchNorm, Dropout, Flatten
- ⚡ **Ativações** — ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- 📉 **Loss Functions** — MSE, CrossEntropy, Binary Cross Entropy
- 🚀 **Optimizers** — SGD (momentum), Adam, RMSProp
- 📦 **Data Loading** — Dataset, DataLoader com batching/shuffling, CSV loader
- 📊 **Métricas** — Accuracy, Precision, Recall, F1-Score
- 💾 **Serialização** — Save/Load de modelos em formato binário
- 📝 **Logger** — Progress bar e logging de treinamento

---

## 🏗️ Arquitetura

```
LunaML/
├── include/lunaml/
│   ├── lunaml.hpp          # Master header (inclui tudo)
│   ├── core/
│   │   ├── types.hpp            # Tipos, aliases, helpers
│   │   ├── tensor.hpp           # Tensor N-dimensional
│   │   ├── tensor_ops.hpp       # Operações sobre tensores
│   │   ├── autograd.hpp         # Diferenciação automática
│   │   └── random.hpp           # Geração de números aleatórios
│   ├── nn/
│   │   ├── module.hpp           # Classe base Module
│   │   ├── linear.hpp           # Camada Dense/Linear
│   │   ├── conv2d.hpp           # Convolução 2D
│   │   ├── pooling.hpp          # MaxPool2D, AvgPool2D
│   │   ├── activation.hpp       # Funções de ativação
│   │   ├── batchnorm.hpp        # Batch Normalization
│   │   ├── dropout.hpp          # Regularização Dropout
│   │   ├── flatten.hpp          # Flatten layer
│   │   └── sequential.hpp       # Container Sequential
│   ├── optim/                   # SGD, Adam, RMSProp
│   ├── loss/                    # MSE, CrossEntropy, BCE
│   ├── data/                    # Dataset, DataLoader, CSV
│   ├── metrics/                 # Accuracy, F1, etc.
│   └── utils/                   # Serialização, Logger
├── src/                         # Implementações (.cpp)
├── tests/                       # Testes unitários (GoogleTest)
└── examples/                    # Exemplos de uso
```

---

## 🚀 Quick Start

### Requisitos

- Compilador C++17 (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.16+

### Build

```bash
# Clone e entre no diretório
git clone <repo-url> LunaML
cd LunaML

# Configure e compile
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)

# Execute os testes
ctest --output-on-failure
```

### Exemplo: Resolvendo XOR

```cpp
#include <lunaml/lunaml.hpp>
using namespace nf;

int main() {
    Random::manual_seed(42);

    // Dados XOR
    auto X = Tensor::from_data({0,0, 0,1, 1,0, 1,1}, {4, 2});
    auto Y = Tensor::from_data({0, 1, 1, 0}, {4, 1});

    // Modelo: 2 -> 16 -> 1
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(2, 16));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(16, 1));
    model.add(std::make_shared<nn::Sigmoid>());

    auto optimizer = optim::Adam(model.parameters(), 0.01f);
    auto criterion = loss::BCE();

    // Loop de treinamento
    for (int epoch = 0; epoch < 1000; ++epoch) {
        auto pred = model.forward(X);
        auto l = criterion(pred, Y);

        optimizer.zero_grad();
        l.backward();
        optimizer.step();

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << " Loss: " << l.item() << "\n";
    }

    // Resultado
    std::cout << "Predictions:\n" << model.forward(X) << "\n";
    return 0;
}
```

---

## 📖 Guia da API

### Tensor

O `Tensor` é o tipo fundamental da LunaML — um array N-dimensional com suporte a autograd.

```cpp
// Criação
auto a = Tensor::zeros({3, 4});           // Tensor 3x4 com zeros
auto b = Tensor::ones({2, 3});            // Tensor 2x3 com uns
auto c = Tensor::randn({5, 5});           // Normal(0, 1)
auto d = Tensor::rand({10, 1});           // Uniforme [0, 1)
auto e = Tensor::from_data({1,2,3,4}, {2, 2});  // Com dados
auto f = Tensor::eye(3);                  // Matriz identidade 3x3

// Operações aritméticas (com autograd)
auto sum = a + b;        // Soma element-wise (com broadcasting)
auto prod = a * b;       // Multiplicação element-wise
auto mm = a.matmul(b);   // Multiplicação de matrizes

// Operações escalares
auto scaled = a * 2.0f;
auto shifted = a + 1.0f;

// Reduções
auto total = a.sum();           // Soma global (escalar)
auto row_sum = a.sum(0);        // Soma ao longo do eixo 0
auto col_mean = a.mean(1, true); // Média com keepdim

// Funções matemáticas
auto ex = a.exp();
auto lg = a.log();
auto pw = a.pow(2.0f);
auto sq = a.sqrt();
auto cl = a.clamp(0.0f, 1.0f);

// Shape
auto reshaped = a.reshape({12});
auto transposed = a.transpose();
auto flat = a.flatten();

// In-place
a.zero_();
a.fill_(3.14f);
a.uniform_(-1.0f, 1.0f);
a.normal_(0.0f, 0.01f);
```

### Autograd

```cpp
// Habilitar gradientes
auto x = Tensor::randn({3, 3}, /*requires_grad=*/true);
auto y = x * x + x * 2.0f;
auto z = y.sum();

// Backpropagation
z.backward();
std::cout << x.grad() << "\n";  // dz/dx = 2x + 2

// Desabilitar gradientes temporariamente
{
    NoGradGuard guard;
    auto no_grad_op = x + x;  // Sem grafo computacional
}
```

### Neural Network Layers

```cpp
// Camada Linear (Dense)
auto linear = nn::Linear(784, 256);  // in=784, out=256
auto output = linear.forward(input);

// Ativações
auto relu = nn::ReLU();
auto sigmoid = nn::Sigmoid();
auto tanh_act = nn::Tanh();
auto softmax = nn::Softmax();
auto leaky = nn::LeakyReLU(0.01f);

// Convolução 2D
auto conv = nn::Conv2d(1, 32, 3, 1, 1);  // in_ch, out_ch, kernel, stride, padding

// Pooling
auto pool = nn::MaxPool2d(2, 2);  // kernel_size, stride

// Regularização
auto dropout = nn::Dropout(0.5f);
auto batchnorm = nn::BatchNorm1d(256);

// Flatten
auto flatten = nn::Flatten();
```

### Sequential

```cpp
nn::Sequential model;
model.add(std::make_shared<nn::Linear>(784, 256));
model.add(std::make_shared<nn::ReLU>());
model.add(std::make_shared<nn::BatchNorm1d>(256));
model.add(std::make_shared<nn::Dropout>(0.3f));
model.add(std::make_shared<nn::Linear>(256, 128));
model.add(std::make_shared<nn::ReLU>());
model.add(std::make_shared<nn::Linear>(128, 10));
model.add(std::make_shared<nn::Softmax>());

// Forward
auto predictions = model.forward(input);

// Parâmetros
auto params = model.parameters();

// Modo treino/avaliação
model.train();  // Ativa dropout/batchnorm treino
model.eval();   // Desativa dropout, usa running stats
```

### Optimizers

```cpp
// SGD com momentum
auto sgd = optim::SGD(model.parameters(), /*lr=*/0.01f, /*momentum=*/0.9f);

// Adam
auto adam = optim::Adam(model.parameters(), /*lr=*/0.001f,
                        /*beta1=*/0.9f, /*beta2=*/0.999f, /*eps=*/1e-8f);

// RMSProp
auto rmsprop = optim::RMSProp(model.parameters(), /*lr=*/0.01f,
                               /*alpha=*/0.99f, /*eps=*/1e-8f);

// Loop de treinamento
optimizer.zero_grad();  // Zera gradientes
loss.backward();        // Calcula gradientes
optimizer.step();       // Atualiza pesos
```

### Loss Functions

```cpp
// Mean Squared Error (regressão)
auto mse = loss::MSE();
auto l = mse(predictions, targets);

// Binary Cross Entropy (classificação binária)
auto bce = loss::BCE();
auto l = bce(predictions, targets);

// Cross Entropy (classificação multi-classe)
auto ce = loss::CrossEntropy();
auto l = ce(predictions, targets);
```

### Data Loading

```cpp
// Dataset personalizado
class MyDataset : public data::Dataset {
public:
    size_t size() const override { return data_.size(); }
    data::Sample get(size_t index) const override {
        return {data_[index], labels_[index]};
    }
private:
    std::vector<Tensor> data_, labels_;
};

// Carregar de CSV
auto dataset = data::CSVDataset("dados.csv", /*target_col=*/0);

// DataLoader com batching e shuffling
auto loader = data::DataLoader(dataset, /*batch_size=*/32, /*shuffle=*/true);

for (auto& [inputs, targets] : loader) {
    auto pred = model.forward(inputs);
    // ...
}
```

### Métricas

```cpp
float acc = metrics::accuracy(predictions, targets);
float prec = metrics::precision(predictions, targets);
float rec = metrics::recall(predictions, targets);
float f1 = metrics::f1_score(predictions, targets);
```

### Serialização

```cpp
// Salvar modelo
utils::save(model, "meu_modelo.nf");

// Carregar modelo
utils::load(model, "meu_modelo.nf");
```

### Logger

```cpp
utils::Logger logger(/*total_epochs=*/100);
logger.log_epoch(epoch, loss_value, accuracy);

// Output:
// [████████████████░░░░░░░░░░░░░░] 53.0% | Epoch 53/100 | Loss: 0.234 | Acc: 91.2%
```

---

## 🧪 Exemplos Incluídos

| Exemplo | Descrição | Arquivo |
|---------|-----------|---------|
| **XOR** | Rede neural resolvendo a porta lógica XOR | `examples/xor_example.cpp` |
| **Regressão** | Regressão linear aprendendo `y = 3x + 2` | `examples/regression.cpp` |

### Executar exemplos

```bash
cd build
./examples/xor_example
./examples/regression
```

---

## 🧪 Testes

A suíte de testes usa GoogleTest (baixado automaticamente via CMake FetchContent).

```bash
cd build
ctest --output-on-failure
```

| Teste | Cobertura |
|-------|-----------|
| `test_tensor` | Criação, operações, shape, indexing, reduções |
| `test_autograd` | Grafo computacional, backward, gradientes |
| `test_layers` | Linear, Conv2D, Pooling, Activation, Sequential |
| `test_optimizers` | SGD, Adam, RMSProp — convergência |
| `test_loss` | MSE, BCE, CrossEntropy |
| `test_integration` | Treino end-to-end com XOR |

---

## 🛣️ Roadmap

- [x] Tensor N-dimensional com broadcasting
- [x] Autograd (diferenciação automática)
- [x] Camadas: Linear, Conv2D, Pooling, BatchNorm, Dropout
- [x] Ativações: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- [x] Optimizers: SGD, Adam, RMSProp
- [x] Loss: MSE, CrossEntropy, BCE
- [x] Data loading com batching
- [x] Métricas e serialização
- [ ] Suporte a RNN/LSTM
- [ ] Aceleração via CUDA/GPU
- [ ] Otimização SIMD (AVX2/SSE)
- [ ] Suporte a fp16/bf16
- [ ] Mais exemplos (MNIST, CIFAR-10)

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

<p align="center">
  Feito com ❤️ e C++ puro.
</p>
