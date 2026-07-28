#pragma once
// GPU acceleration hooks, only compiled/linked in when the project is built
// with -DUSE_CUDA (see Makefile's `cuda` target, which builds with nvcc).
// The CPU build (`make cpu`, plain g++) never sees these declarations and
// keeps running the original hand-written loops in Tensor.cpp / Linear.cpp.
#include <cstddef>

#ifdef USE_CUDA

// Batched matmul: C = A @ B
//   A: [batch, M, K] contiguous row-major
//   B: [batch, K, N] contiguous row-major
//   C: [batch, M, N] contiguous row-major (already allocated by caller)
// Used by Tensor::matmul, which is the primitive ScaledDotProductAttention
// builds Q@K^T and softmax(scores)@V out of.
void cuda_batched_matmul(const double* A, const double* B, double* C, size_t batch, size_t M, size_t K, size_t N);

// Linear layer forward: C[m,n] = bias[n] + sum_k A[m,k] * W[n,k]
//   A: (M, K)   -- M = seqLen, K = inputSize
//   W: (N, K)   -- N = outputSize (PyTorch-style weight layout, same as this
//                  project's Linear layer: rows are output units)
//   bias: (N)
//   C: (M, N) (already allocated by caller)
void cuda_linear_forward(const double* A, const double* W, const double* bias, double* C, size_t M, size_t K, size_t N);

// Linear layer backward.
//   dOut : (M, N)  upstream gradient
//   A    : (M, K)  cached forward input (last_input)
//   W    : (N, K)  weights
//   dW   : (N, K)  ACCUMULATED in place (caller must seed with current grad)
//   dBias: (N)     ACCUMULATED in place (caller must seed with current grad)
//   dA   : (M, K)  written (not accumulated) -- this call's input gradient
void cuda_linear_backward(const double* dOut, const double* A, const double* W, double* dW, double* dBias, double* dA, size_t M, size_t K, size_t N);

#endif // USE_CUDA
