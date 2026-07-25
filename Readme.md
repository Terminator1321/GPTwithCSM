# GPTwithCSM

A GPT/Llama-style transformer, built entirely from scratch in **Modern C++20** — no PyTorch, TensorFlow, Eigen, Armadillo, xtensor, or any ML library. This is the upgrade path from [CSM](https://github.com/Terminator1321/CSM-), a classic feedforward neural network also written in raw C++, toward a full transformer architecture.

## Philosophy

Everything — tensors, gradients, layers, optimizers, attention, the training loop — is implemented manually. The design borrows ideas from PyTorch (a `Tensor`/`Parameter`/`Module` split) and llama.cpp (contiguous, cache-friendly memory), but stays small enough to read end to end and understand every line of math it runs.

Core principles the code follows:
- **Contiguous memory** — tensor storage is a single flat buffer, never nested `vector<vector<...>>`.
- **Manual gradients, no autograd graph** — every layer computes its own backward pass by hand; there is no automatic differentiation engine tracking an op graph.
- **Separation of computation and update** — layers only *accumulate* gradients into a `Parameter`; only an optimizer is allowed to modify a `Parameter`'s value. No layer applies a learning rate itself.
- **Views over copies** — reshape, transpose, permute, and slice operations share the underlying buffer instead of copying data, wherever the operation allows it.

## Architecture

```
gpt/
  Tensor/
    Tensor.hpp / Tensor.cpp        - n-dimensional array: shape, strides, views,
                                      broadcasting, batched matmul, reductions,
                                      gradient buffer, Xavier/He initializers

  core/
    Parameter.hpp                  - value + grad + trainable, the unit every
                                      optimizer updates
    Module.hpp                     - shared parameters()/train()/eval() interface
                                      implemented by every layer

  Layers/
    Embedding.hpp / Embedding.cpp  - trainable token embedding table with batch
                                      lookup and gradient-accumulating backward
    Linear.hpp / Linear.cpp        - fully connected layer (forward implemented;
                                      full backward still pending, see below)

  TokenizerLayer/
    Tokenizer.hpp / Tokenizer.cpp  - character-level tokenizer: builds a
                                      vocabulary from raw text files and maps
                                      characters to/from token indices

  helper/
    RandomWeights.hpp              - lightweight uniform weight sampler

Data/                              - raw text corpora used to build the vocabulary
main.cpp                           - entry point (currently runs tokenization)
```

## Current status

| Component | Status |
|---|---|
| Tensor | ✅ Done — arbitrary rank, strides, views, broadcasting, batched matmul, reductions, gradient buffer, Xavier/He init |
| Parameter | ✅ Done — `value` / `grad` / `trainable` |
| Module | ✅ Done — shared `parameters()` / `train()` / `eval()` / `zero_grad()` |
| Embedding | ✅ Done — batch lookup, gradient accumulation, no in-layer parameter updates |
| Linear | ⏳ Forward pass only — `dWeight`/`dBias` not yet computed or stored |
| LayerNorm | ⬜ Not started |
| Softmax | ⬜ Not started |
| Dropout | ⬜ Not started |
| GELU | ⬜ Not started |
| CrossEntropy | ⬜ Not started |
| Optimizers (SGD, Momentum SGD, Adam, AdamW) | ⬜ Not started |
| Attention | ⬜ Not started |
| MultiHeadAttention | ⬜ Not started |
| FeedForward | ⬜ Not started |
| TransformerBlock | ⬜ Not started |
| GPT (full model) | ⬜ Not started |
| Training loop | ⬜ Not started |
| Tokenizer | ✅ Done — character-level, vocabulary build + save/load |

## Requirements

- A C++20-capable compiler (GCC 10+, Clang 12+, or MSVC 2019 16.10+)
- No external dependencies — the entire framework is self-contained

## Related repository

[CSM](https://github.com/Terminator1321/CSM-) — the original, non-transformer neural network this project evolved from.