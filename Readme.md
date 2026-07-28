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
    Parameter.hpp                  - value + grad + Adam moments (m, v) + trainable,
                                      the unit every optimizer updates
    Module.hpp                     - shared parameters()/train()/eval()/zero_grad()
                                      interface implemented by every layer

  Layers/
    Embedding.hpp / Embedding.cpp  - trainable token embedding table with batch
                                      lookup and gradient-accumulating backward
    PositionEmbedding.hpp / .cpp   - learned positional embedding table, forward
                                      (slice) + backward (accumulate)
    Linear.hpp / Linear.cpp        - fully connected layer, forward + backward
                                      (dWeight, dBias, dInput), 1D and 2D input
    LayerNorms.hpp / .cpp          - layer normalization (gamma, beta affine),
                                      forward + backward
    ScaledDotProductAttention.*    - the per-head attention computation (causal
                                      mask, softmax), forward + backward
    MultiHeadAttention.hpp / .cpp  - Q/K/V/O projections + scaled dot-product
                                      attention, split across heads, forward +
                                      backward
    MLP.hpp / MLP.cpp              - position-wise feed-forward block (GELU),
                                      forward + backward

  NameSpaces/
    ActivationFunction.hpp / .cpp  - Softmax and GELU, forward + backward
    LOSS.hpp / .cpp                - Cross-entropy loss, forward + backward

  Optimizers/
    Optimizer.hpp                  - abstract base: per-parameter step() + a
                                      batch step() over parameters() that also
                                      advances a shared timestep for bias
                                      correction
    Adam.hpp / Adam.cpp            - Adam optimizer
    AdamW.hpp / AdamW.cpp          - AdamW optimizer (decoupled weight decay)

  Transformer.hpp / Transformer.cpp - one pre-LN transformer block: LN1 ->
                                      attention -> residual -> LN2 -> MLP ->
                                      residual, forward + backward

  GPT.hpp / GPT.cpp                - full model: token + position embedding,
                                      a stack of transformer blocks, final
                                      LayerNorm, LM head, full backward(),
                                      trainOnBatch() (forward + loss + backward
                                      in one call), sampling-based generate(),
                                      and summary()

  ModelSummary.hpp / .cpp          - PyTorch/TensorFlow-style architecture
                                      table printer, called via `GPT::summary()`

  TokenizerLayer/
    Tokenizer.hpp / Tokenizer.cpp  - character-level tokenizer: builds a
                                      vocabulary from raw text files, maps
                                      characters to/from token indices, and
                                      saves/loads the vocabulary to a file

  helper/
    RandomWeights.hpp              - lightweight uniform weight sampler
    mask.hpp                       - causal (look-ahead) attention mask
    Sampling.hpp                   - temperature + top-k sampling for generate()

Data/                              - raw text corpora used to build the vocabulary
main.cpp                           - entry point: tokenizes the corpus, builds a
                                      small GPT, prints its architecture summary,
                                      trains it for a fixed number of epochs with
                                      Adam, then drops into an interactive prompt
                                      loop that calls generate()
```

## Building

There's no CMake project yet — a single compiler invocation builds the whole thing:

```bash
g++ -std=c++20 -I. main.cpp gpt/GPT.cpp gpt/Transformer.cpp gpt/ModelSummary.cpp \
    gpt/Layers/*.cpp gpt/Tensor/*.cpp \
    gpt/NameSpaces/ActivationFunction.cpp gpt/NameSpaces/LOSS.cpp \
    gpt/Optimizers/Adam.cpp gpt/Optimizers/AdamW.cpp \
    gpt/TokenizerLayer/*.cpp \
    -o gpt_main
```

Run `./gpt_main` (or `gpt_main.exe` on Windows) from the project root so the relative `./Data/...` paths resolve correctly.

## Model Summary

`GPT` exposes a `model.summary()`-style method:

```cpp
GPT gpt(vocabSize, embedDim, maxSeqLen, numHeads, numLayers);
gpt.summary();
```

This prints a Unicode box-drawing table (one row per layer, every transformer
block expanded into its LayerNorm / Attention / Residual / MLP / Residual
sub-layers) followed by total/trainable/non-trainable parameter counts,
estimated memory, and the model's hyperparameters. The parameter counts are
computed directly from the real layer shapes (`Linear` = `out*in + out`,
`LayerNorm` = `2*dim`, embeddings have no bias, etc.) — not hardcoded — so
the table stays correct for any `numLayers`/`embedDim`/`numHeads` combination.
Memory is reported in FP64, matching `Tensor`'s `double`-backed storage.
`main.cpp` calls this automatically right after constructing its GPT
instance.

## Current status

Every layer in the forward path now has a matching, hand-derived backward
pass, and they are all wired together end to end: `GPT::backward()` walks
lm-head to final LayerNorm to transformer blocks (in reverse) to position
embedding to token embedding, and `GPT::trainOnBatch()` runs forward, softmax
plus cross-entropy, and backward in one call. `main.cpp` already trains a
small model with Adam for a fixed number of epochs and reports train/
validation loss per epoch, then serves an interactive `generate()` prompt
loop.

| Component | Status |
|---|---|
| Tensor | Done — arbitrary rank, strides, views, broadcasting, batched matmul, reductions, gradient buffer, Xavier/He init |
| Parameter | Done — `value` / `grad` / Adam moments (`m`, `v`) / `trainable` |
| Module | Done — shared `parameters()` / `train()` / `eval()` / `zero_grad()` |
| Embedding | Done — forward + backward (gradient accumulation per token id) |
| PositionEmbedding | Done — forward (slice) + backward (accumulate) |
| Linear | Done — forward + backward (`dWeight`, `dBias`, `dInput`), 1D and 2D inputs |
| LayerNorm | Done — forward + full backward (`dgamma`, `dbeta`, `dx`) |
| Softmax | Done — forward + backward |
| GELU | Done — forward + backward (tanh approximation) |
| CrossEntropy | Done — forward + backward |
| Scaled Dot-Product Attention | Done — forward (causal mask) + backward (`dQ`, `dK`, `dV`) |
| MultiHeadAttention | Done — forward (split heads) + backward (concat + merge heads) |
| FeedForward (MLP) | Done — forward + backward through both Linear layers and GELU |
| TransformerBlock | Done — forward + backward through the full pre-LN residual block |
| GPT (full model) | Done — forward, backward, `trainOnBatch()`, sampling `generate()` (temperature + top-k), `summary()` |
| Optimizers (Adam, AdamW) | Done |
| Training loop | Basic version done in `main.cpp` — one sequence at a time, fixed epoch count, train/val loss reporting. Still needed: true mini-batching, gradient clipping, LR scheduling |
| Model checkpointing (save/load) | Not started — no way yet to serialize/restore weights, optimizer state, or hyperparameters |
| Dropout | Not started |
| KV cache for `generate()` | Not started — every generation step reruns the full forward pass |
| CUDA backend | Not started, planned as an optional addition |
| Tokenizer | Done — character-level, vocabulary build + save/load |

## What's left

The remaining work, roughly in priority order:

1. **Model saving & loading (checkpointing)** — the main gap right now. This means:
   - Serializing every `Parameter`'s `value` (and, for a resumable checkpoint, its Adam `m`/`v` moments) to a binary or text file, in the same order `GPT::parameters()` returns them.
   - Serializing the `GPTConfig` (`vocabSize`, `embedDim`, `maxSeqLen`, `numHeads`, `numLayers`) alongside the weights so a checkpoint is self-describing and can reconstruct the exact same architecture before loading weights into it.
   - Serializing the tokenizer vocabulary (`Tokenizer` already has `SaveTokensToFile`/`LoadTokensFromFile` — this just needs to be bundled with the model checkpoint).
   - A `save(path)` / `load(path)` pair on `GPT` (and ideally on `Optimizer`, so training can resume exactly, including bias-correction timesteps) so training doesn't have to restart from scratch every run.
2. **Training loop hardening** — real mini-batch training (currently each sequence is a separate forward/backward/step, effectively batch size 1), gradient clipping to guard against exploding gradients, and a learning-rate schedule (warmup + decay, standard for transformers).
3. **CUDA backend (optional)** — move `Tensor`'s batched matmul and elementwise ops onto the GPU behind a compile-time flag, keeping the existing CPU path as the default/fallback so the project still builds and runs with nothing but a C++20 compiler.
4. Further out: dropout, KV-cache-accelerated `generate()`, gradient checkpointing for memory, and a proper BPE/subword tokenizer to replace the character-level one.

## Requirements

- A C++20-capable compiler (GCC 10+, Clang 12+, or MSVC 2019 16.10+)
- No external dependencies — the entire framework is self-contained

## Related repository

[CSM](https://github.com/Terminator1321/CSM-) — the original, non-transformer neural network this project evolved from.

---

# Mathematical Notes

This section documents the exact math each module implements, forward and
backward, in the order data flows through the model:
`Tensor ops -> Embedding -> PositionEmbedding -> LayerNorm -> Linear ->
Scaled Dot-Product Attention -> MultiHeadAttention -> GELU -> FeedForward
(MLP) -> TransformerBlock -> GPT -> Softmax + Cross-Entropy -> Adam / AdamW`.

Notation used throughout:
- $L$ — the scalar loss.
- $\frac{\partial L}{\partial z}$ — the "upstream gradient" arriving at a
  module from the layer *after* it; every backward function receives this
  and must return $\frac{\partial L}{\partial x}$ for its own input $x$
  (and accumulate gradients into its own parameters, if it has any).
- $x_i, y_i$ — the $i$-th component of a vector $x$ or $y$.
- $H$ — the embedding/hidden dimension (`embedDim`).
- $N$ — sequence length (number of tokens in the current forward pass).
- $V$ — vocabulary size.

Every layer follows the same contract: `forward(x)` caches whatever
intermediate values its `backward(dOut)` will need, and `backward` returns
`dInput` while accumulating into any `Parameter::grad` it owns — it never
touches `Parameter::value` directly. Only the optimizer does that.

## Tensor: batched matmul and broadcasting

`Tensor::matmul(A, B)` treats the last two dimensions of `A` and `B` as
matrices and every leading dimension as an independent batch, so a single
call computes attention scores for every head in one pass. For a single
matrix pair, standard matmul and its gradient are:

$$C = AB \qquad C_{ij} = \sum_k A_{ik} B_{kj}$$

Given an upstream gradient $\frac{\partial L}{\partial C}$:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \, B^\top
\qquad
\frac{\partial L}{\partial B} = A^\top \, \frac{\partial L}{\partial C}$$

Every backward pass in this project that involves a matmul (`Linear`,
`ScaledDotProductAttention`) is a direct application of these two rules,
computed manually rather than through an autograd graph. Broadcasting (e.g.
adding a bias vector to every row) follows the usual rule that the gradient
with respect to the broadcast operand is the sum over the broadcast axes.

## Embedding

**Forward.** The embedding table is a weight matrix $W_{emb} \in
\mathbb{R}^{V \times H}$. For a token id $t$, the embedding is simply the
$t$-th row:

$$y_s = W_{emb}[t_s], \qquad s = 0, \dots, N-1$$

Because a lookup is a discrete operation, there is no computation to
differentiate through for the *value* being looked up — only for the table
itself.

**Backward.** Given the upstream gradient $\frac{\partial L}{\partial y_s}$
for every position $s$, each row of the table only receives gradient
contributions from the positions where it was actually used, summed
(accumulated) if a token id repeats in the sequence:

$$\frac{\partial L}{\partial W_{emb}[i]} = \sum_{s \,:\, t_s = i} \frac{\partial L}{\partial y_s}$$

Rows for token ids that never appeared in the batch get no gradient at all
this step. There is no gradient with respect to the token ids themselves —
`Embedding::backward` returns nothing to "earlier" layers because the input
(a `vector<int>`) is not differentiable.

## PositionEmbedding

**Forward.** A second learned table $P \in \mathbb{R}^{\text{maxSeqLen}
\times H}$ stores one vector per absolute position. For a sequence of
length $N$, the forward pass is just the first $N$ rows:

$$y_s = P[s], \qquad s = 0, \dots, N-1$$

which is implemented as a zero-copy `slice`, not a data copy.

**Backward.** Identical accumulation rule to Embedding, but by position
instead of by token id (positions don't repeat within one sequence, so no
summing collision happens in practice, but the code handles it generally):

$$\frac{\partial L}{\partial P[s]} \mathrel{+}= \frac{\partial L}{\partial y_s}$$

The token and position embeddings are summed elementwise before entering the
first transformer block ($x = W_{emb}[t] + P[\text{pos}]$), so their
backward passes both receive the *same* upstream gradient independently —
addition simply routes the identical gradient to both branches.

## Linear (fully connected layer)

**Forward.** For weight matrix $W \in \mathbb{R}^{\text{out}\times\text{in}}$
and bias $b \in \mathbb{R}^{\text{out}}$, applied per token:

$$y = Wx + b, \qquad y_o = b_o + \sum_i W_{oi}\, x_i$$

**Backward.** Given $\frac{\partial L}{\partial y}$, the three gradients are:

$$\frac{\partial L}{\partial b_o} = \frac{\partial L}{\partial y_o}
\qquad
\frac{\partial L}{\partial W_{oi}} = \frac{\partial L}{\partial y_o} \cdot x_i
\qquad
\frac{\partial L}{\partial x_i} = \sum_o \frac{\partial L}{\partial y_o} \cdot W_{oi}$$

The bias gradient is a direct pass-through of the upstream gradient, the
weight gradient is the outer product of the upstream gradient and the
cached input (`last_input`), and the input gradient is the upstream
gradient multiplied by $W$ — exactly the batched-matmul rule above, applied
per token when the input is a `(seqLen, in)` tensor. All three are
accumulated into `weights.grad` / `bias.grad`, never applied to
`weights.value` / `bias.value` directly.

## LayerNorm

**Forward.** For each token's feature vector $x = [x_1, \dots, x_H]$:

$$\mu = \frac{1}{H}\sum_{i=1}^H x_i
\qquad
\sigma^2 = \frac{1}{H}\sum_{i=1}^H (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\qquad
y_i = \gamma_i \hat{x}_i + \beta_i$$

with $\epsilon = 10^{-5}$ for numerical stability and $\gamma, \beta \in
\mathbb{R}^H$ learned per-feature scale/shift. The normalized value
$\hat{x}$ and $1/\sqrt{\sigma^2+\epsilon}$ are cached for backward.

**Backward.** The parameter gradients are straightforward sums over the
upstream gradient $\frac{\partial L}{\partial y}$:

$$\frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i}\cdot\hat{x}_i
\qquad
\frac{\partial L}{\partial \beta_i} = \frac{\partial L}{\partial y_i}$$

The input gradient is the standard LayerNorm backward formula, which
accounts for the fact that $\mu$ and $\sigma^2$ both depend on every $x_i$
(so each output depends on every input, not just the matching index):

$$\frac{\partial L}{\partial x_i} = \frac{1}{H\sqrt{\sigma^2+\epsilon}}
\left[ H\, g_i - \sum_{j=1}^H g_j - \hat{x}_i \sum_{j=1}^H g_j \hat{x}_j \right],
\qquad g_i \equiv \frac{\partial L}{\partial y_i}\cdot\gamma_i$$

In the code this is computed in one pass per token: accumulate
`dnorm_sum` ($\sum g_j$) and `dnorm_dot_norm` ($\sum g_j \hat{x}_j$), then
apply the bracketed expression scaled by `inv_std / H` for every feature.

**Worked example.** With $x = [1.0, 3.0, 4.0, 6.0]$, $\gamma=1,\ \beta=0$:
$\mu = 3.5$, $\sigma^2 = 3.25$, $\sqrt{\sigma^2+\epsilon}\approx 1.8028$, giving
$\hat{x} \approx [-1.3867, -0.2773, 0.2773, 1.3867]$. For an assumed
upstream gradient $\frac{\partial L}{\partial y} = [0.5, -0.2, 0.1, 0.4]$:
$\frac{\partial L}{\partial \beta} = 0.8$,
$\frac{\partial L}{\partial \gamma} \approx -0.0555$, and working through the
bracket formula for each feature gives
$\frac{\partial L}{\partial x} \approx [0.1558,\, -0.2241,\, -0.0533,\, 0.1216]$.

## GELU

**Forward.** This project uses the tanh-based GELU approximation (the same
one GPT-2 uses):

$$\text{GELU}(x) = 0.5\,x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right]\right)$$

**Backward.** Let $u(x) = \sqrt{2/\pi}(x + 0.044715 x^3)$ and $t = \tanh(u)$.
Differentiating the forward expression with respect to $x$ (product rule +
chain rule through $\tanh$, using $\frac{d}{dx}\tanh(u) = (1-t^2)u'(x)$):

$$\text{GELU}'(x) = 0.5(1+t) + 0.5\,x\,(1-t^2)\sqrt{2/\pi}\,(1 + 3\cdot 0.044715\,x^2)$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\cdot \text{GELU}'(x)$$

which matches `geluBackward` exactly — it recomputes $u$, $t$, $1-t^2$ from
the cached pre-activation input and multiplies by the upstream gradient.

## Softmax

**Forward.** For a row of logits $z = [z_1, \dots, z_K]$ (numerically
stabilized by subtracting the row max before exponentiating):

$$p_i = \frac{e^{z_i - \max_j z_j}}{\sum_{k} e^{z_k - \max_j z_j}}$$

**Backward.** Softmax's Jacobian is dense — every output depends on every
input — so the gradient with respect to $z_i$ needs a sum over all outputs:

$$\frac{\partial L}{\partial z_i} = p_i\left(\frac{\partial L}{\partial p_i} - \sum_k \frac{\partial L}{\partial p_k}\, p_k\right)$$

The code computes the dot product $\sum_k \frac{\partial L}{\partial p_k} p_k$
once per row, then applies this formula per element — this is the standard
"softmax times (grad minus weighted-sum-of-grad)" identity, and it's reused
by both the loss softmax in `trainOnBatch` and the attention softmax inside
`ScaledDotProductAttention`.

## Cross-Entropy Loss

**Forward.** For predicted probabilities $p$ (already the output of
softmax) and a target class index $t$:

$$L = -\log(p_t)$$

$p_t$ is clamped to $[10^{-12}, 1-10^{-12}]$ first so the log never
overflows on a perfectly confident (or perfectly wrong) prediction.

**Backward.** Cross-entropy alone (with respect to $p$, not the logits) is
sparse — only the target class receives a nonzero gradient:

$$\frac{\partial L}{\partial p_i} = \begin{cases} -1/p_t & i = t \\ 0 & i \neq t \end{cases}$$

In practice the model fuses softmax and cross-entropy: `trainOnBatch`
builds `dLogits` directly as $p - \text{one\_hot}(t)$, which is the well
known simplified gradient of softmax-then-cross-entropy with respect to the
*logits* (derived by substituting the softmax backward formula into the
cross-entropy backward formula and simplifying — the dense softmax Jacobian
collapses to this single subtraction). This is more numerically stable and
far cheaper than composing the two backward passes separately.

## Scaled Dot-Product Attention

**Forward.** For per-head queries, keys, values $Q, K, V \in
\mathbb{R}^{N \times d_k}$:

$$\text{scores} = \frac{QK^\top}{\sqrt{d_k}}$$

A causal mask then sets every entry above the diagonal (positions attending
to the future) to $-\infty$ before softmax, so a token can only attend to
itself and earlier tokens:

$$A = \text{softmax}(\text{scores}_{\text{masked}}) \qquad \text{Output} = AV$$

**Backward.** Given $\frac{\partial L}{\partial \text{Output}}$, backprop
through $\text{Output}=AV$ using the matmul rule:

$$\frac{\partial L}{\partial V} = A^\top \frac{\partial L}{\partial \text{Output}}
\qquad
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \text{Output}}\, V^\top$$

Then through the softmax (using the softmax backward rule above, row-wise
over the masked scores) to get $\frac{\partial L}{\partial \text{scores}_{\text{masked}}}$,
rescale by $1/\sqrt{d_k}$, and backprop through $\text{scores}=QK^\top$:

$$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial \text{scores}}\, K
\qquad
\frac{\partial L}{\partial K} = \left(\frac{\partial L}{\partial \text{scores}}\right)^{\!\top} Q$$

The masked (-inf) positions produce exactly zero softmax probability in the
forward pass, so their gradient contribution during backward is
automatically zero as well — no separate masking step is needed on the way
back.

## MultiHeadAttention

**Forward.** Input $x \in \mathbb{R}^{N\times H}$ is projected to queries,
keys, and values with three independent `Linear` layers:

$$Q = W_qx,\quad K=W_kx,\quad V=W_vx$$

Each is reshaped from $(N, H)$ into $(\text{numHeads}, N, d_k)$ where
$d_k = H/\text{numHeads}$, so every head gets an independent
`ScaledDotProductAttention` over its own slice of the embedding. The
per-head outputs are concatenated back into $(N, H)$ and passed through a
final output projection:

$$\text{Output} = W_o \cdot \text{concat}(\text{head}_1, \dots, \text{head}_{\text{numHeads}})$$

**Backward.** This is purely bookkeeping around the pieces already derived
above: backprop through $W_o$ (`Linear` backward) to get the gradient with
respect to the concatenated heads, split that back into per-head gradients,
run each head's `ScaledDotProductAttention::backward` to get $dQ, dK, dV$
per head, reassemble/reshape them back into $(N, H)$, run each of
$W_q, W_k, W_v$'s `Linear::backward` on its respective gradient, and sum the
three resulting input gradients — because $Q$, $K$, and $V$ all came from
the *same* input $x$ via three separate branches, their gradients add:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial x}\Big|_{Q} + \frac{\partial L}{\partial x}\Big|_{K} + \frac{\partial L}{\partial x}\Big|_{V}$$

## FeedForward (MLP)

**Forward.** A standard two-layer position-wise network with GELU in
between, expanding to $4H$ hidden units:

$$h = \text{GELU}(W_1 x + b_1) \qquad y = W_2 h + b_2$$

**Backward.** Straight chain rule through the three pieces, each already
derived above:

$$\frac{\partial L}{\partial x} = W_1^\top\Big[\text{GELU}'(W_1x+b_1)\odot\big(W_2^\top\, \tfrac{\partial L}{\partial y}\big)\Big]$$

implemented in code as `fc2.backward` -> `geluBackward` (using the cached
pre-GELU activation `fc1_out`) -> `fc1.backward`, each accumulating into
their own `weights.grad`/`bias.grad` along the way.

## TransformerBlock (pre-LN residual block)

**Forward.** One block is: normalize, attend, add back the un-normalized
input (residual), normalize again, feed-forward, add back again:

$$h = x + \text{Attention}(\text{LN}_1(x)) \qquad y = h + \text{MLP}(\text{LN}_2(h))$$

**Backward.** Because addition routes an identical copy of the upstream
gradient to *both* branches feeding into it, each residual connection splits
the gradient rather than merging it — so the gradient with respect to $h$
has two contributions (one straight through the residual, one through the
MLP branch), and likewise for $x$:

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial y} + \text{LN}_2\text{-backward}\!\left(\text{MLP-backward}\!\left(\frac{\partial L}{\partial y}\right)\right)$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h} + \text{LN}_1\text{-backward}\!\left(\text{Attention-backward}\!\left(\frac{\partial L}{\partial h}\right)\right)$$

This is exactly what `TransformerBlock::backward` does: it keeps a clone of
the incoming gradient as the "skip" term, adds the gradient coming back
through the normalized branch, and does this once for each of the two
residual connections in the block.

## GPT (full model)

**Forward.** Token and position embeddings are summed, passed through every
`TransformerBlock` in order, normalized once more, then projected to
vocabulary logits by the LM head:

$$x = W_{emb}[t] + P[\text{pos}] \;\;\to\;\; \text{Block}_1 \to \cdots \to \text{Block}_L \;\;\to\;\; \text{LN}_{\text{final}} \;\;\to\;\; \text{logits} = W_{\text{head}}\, x$$

**Backward.** The exact reverse order — this is the top-level chain rule
that ties every module above together:

$$\text{logits} \;\to\; W_{\text{head}}\text{-backward} \;\to\; \text{LN}_{\text{final}}\text{-backward} \;\to\; \text{Block}_L\text{-backward} \to \cdots \to \text{Block}_1\text{-backward} \;\to\; \{\text{PositionEmbedding, Embedding}\}\text{-backward}$$

`GPT::backward()` implements exactly this: it walks blocks in reverse with
a C++ reverse iterator, and because the token and position embeddings were
summed (not chained) at the start, both receive the *same* final gradient
independently, following the addition rule described earlier.
`GPT::trainOnBatch()` chains everything together per call: forward, then
softmax plus cross-entropy per token (with the fused $p - \text{one\_hot}(t)$
gradient described above), then `backward()` — one full training step for
one sequence.

## Adam and AdamW

Both optimizers only ever touch `Parameter::value`; every gradient they
consume was produced purely by the backward passes above, and `zero_grad()`
must be called between steps since gradients are accumulated, not
overwritten.

**Adam.** Maintains exponential moving averages of the gradient (`m`, first
moment) and its square (`v`, second moment), then bias-corrects both before
the update (bias correction matters most in early steps, since `m`/`v`
start at zero):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

with defaults $\eta=10^{-3}, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$.
`t` is the shared `timestep_` on `Optimizer`, incremented once per call to
the batch `step(params)` overload (not once per parameter), so every
parameter in the model stays on the same bias-correction schedule.

**AdamW.** Identical moment estimates and bias correction, but weight decay
is applied directly to the *weights* rather than folded into the gradient
(the "decoupled" part of AdamW, which avoids interacting with Adam's
per-parameter adaptive scaling):

$$\theta_t = \theta_{t-1} - \eta\,\lambda\,\theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

with an added weight-decay coefficient $\lambda$ (default $0.01$). This
matches `AdamW::step`, which subtracts the decay term before the usual Adam
update.

## Sampling (`generate()`)

Not a gradient path — this only runs at inference time inside
`GPT::generate()` — but it's worth documenting since it's the last step of
the pipeline. Given the logits for the final position:

1. **Temperature scaling:** divide every logit by `temperature` before
   softmax (`temperature < 1` sharpens the distribution toward the argmax,
   `> 1` flattens it toward uniform).
2. **Top-k filtering:** keep only the `top_k` highest-scoring logits, set
   everything else to $-\infty$ so it can never be sampled.
3. **Softmax** the (possibly filtered) logits into a probability
   distribution, numerically stabilized the same way as everywhere else in
   the project.
4. **Sample** a token id from that categorical distribution using
   `std::discrete_distribution`, append it to the sequence, and repeat.