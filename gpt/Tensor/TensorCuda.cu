#include "TensorCuda.cuh"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

namespace {

constexpr int kBlock = 16;
constexpr int kBlock1D = 256;

// C[b] = A[b] @ B[b], row-major, one thread per output element.
__global__ void batchedMatmulKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, size_t M, size_t K, size_t N)
{
    size_t batch = blockIdx.z;
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
        return;

    const double* Ab = A + batch * M * K;
    const double* Bb = B + batch * K * N;
    double sum = 0.0;
    for (size_t k = 0; k < K; ++k)
        sum += Ab[row * K + k] * Bb[k * N + col];
    C[batch * M * N + row * N + col] = sum;
}

// C[m,n] = bias[n] + sum_k A[m,k] * W[n,k]   (W rows are output units, so
// this is effectively A @ W^T, matching the project's (out,in) weight layout)
__global__ void linearForwardKernel(const double* __restrict__ A,
                                     const double* __restrict__ W,
                                     const double* __restrict__ bias,
                                     double* __restrict__ C,
                                     size_t M, size_t K, size_t N)
{
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y; // token
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; // out unit
    if (row >= M || col >= N)
        return;

    const double* Arow = A + row * K;
    const double* Wrow = W + col * K;
    double sum = bias[col];
    for (size_t k = 0; k < K; ++k)
        sum += Arow[k] * Wrow[k];
    C[row * N + col] = sum;
}

// dW[out,in] += sum_token dOut[token,out] * A[token,in]
__global__ void linearBackwardWeightsKernel(const double* __restrict__ dOut, const double* __restrict__ A, double* __restrict__ dW, size_t M, size_t K, size_t N)
{
    size_t out = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t in = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= N || in >= K)
        return;

    double sum = 0.0;
    for (size_t token = 0; token < M; ++token)
        sum += dOut[token * N + out] * A[token * K + in];
    dW[out * K + in] += sum;
}

// dBias[out] += sum_token dOut[token,out]
__global__ void linearBackwardBiasKernel(const double* __restrict__ dOut, double* __restrict__ dBias, size_t M, size_t N)
{
    size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= N)
        return;

    double sum = 0.0;
    for (size_t token = 0; token < M; ++token)
        sum += dOut[token * N + out];
    dBias[out] += sum;
}

// dA[token,in] = sum_out dOut[token,out] * W[out,in]
__global__ void linearBackwardInputKernel(const double* __restrict__ dOut, const double* __restrict__ W, double* __restrict__ dA, size_t M, size_t K, size_t N)
{
    size_t token = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t in = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= M || in >= K)
        return;

    double sum = 0.0;
    for (size_t out = 0; out < N; ++out)
        sum += dOut[token * N + out] * W[out * K + in];
    dA[token * K + in] = sum;
}

} // namespace

void cuda_batched_matmul(const double* A, const double* B, double* C, size_t batch, size_t M, size_t K, size_t N)
{
    if (batch == 0 || M == 0 || K == 0 || N == 0)
        return;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytesA = batch * M * K * sizeof(double);
    const size_t bytesB = batch * K * N * sizeof(double);
    const size_t bytesC = batch * M * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(kBlock, kBlock);
    dim3 grid(static_cast<unsigned>((N + kBlock - 1) / kBlock),
               static_cast<unsigned>((M + kBlock - 1) / kBlock),
               static_cast<unsigned>(batch));
    batchedMatmulKernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void cuda_linear_forward(const double* A, const double* W, const double* bias, double* C, size_t M, size_t K, size_t N)
{
    double *dA = nullptr, *dW = nullptr, *dBias = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dW, N * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dBias, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dA, A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, W, N * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, bias, N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(kBlock, kBlock);
    dim3 grid(static_cast<unsigned>((N + kBlock - 1) / kBlock),
               static_cast<unsigned>((M + kBlock - 1) / kBlock));
    linearForwardKernel<<<grid, block>>>(dA, dW, dBias, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dBias);
    cudaFree(dC);
}

void cuda_linear_backward(const double* dOut, const double* A, const double* W, double* dW, double* dBias, double* dA, size_t M, size_t K, size_t N)
{
    double *d_dOut = nullptr, *d_A = nullptr, *d_W = nullptr;
    double *d_dW = nullptr, *d_dBias = nullptr, *d_dA = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dOut, M * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, N * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dW, N * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dBias, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dA, M * K * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_dOut, dOut, M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, N * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dW, dW, N * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dBias, dBias, N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blockW(kBlock, kBlock);
    dim3 gridW(static_cast<unsigned>((K + kBlock - 1) / kBlock),
                static_cast<unsigned>((N + kBlock - 1) / kBlock));
    linearBackwardWeightsKernel<<<gridW, blockW>>>(d_dOut, d_A, d_dW, M, K, N);
    CUDA_CHECK(cudaGetLastError());

    unsigned gridB = static_cast<unsigned>((N + kBlock1D - 1) / kBlock1D);
    linearBackwardBiasKernel<<<gridB, kBlock1D>>>(d_dOut, d_dBias, M, N);
    CUDA_CHECK(cudaGetLastError());

    dim3 blockA(kBlock, kBlock);
    dim3 gridA(static_cast<unsigned>((K + kBlock - 1) / kBlock),
                static_cast<unsigned>((M + kBlock - 1) / kBlock));
    linearBackwardInputKernel<<<gridA, blockA>>>(d_dOut, d_W, d_dA, M, K, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dW, d_dW, N * K * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dBias, d_dBias, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dA, d_dA, M * K * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_dOut);
    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_dW);
    cudaFree(d_dBias);
    cudaFree(d_dA);
}

#endif // USE_CUDA
