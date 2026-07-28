// Standalone benchmark: times Tensor::matmul (the op CUDA accelerates) at a
// few representative sizes, CPU vs (if built with -DUSE_CUDA) GPU.
//
//   g++      -std=c++20 -O2 -I. benchmark_matmul.cpp gpt/Tensor/Tensor.cpp -o bench_cpu
//   nvcc     -std=c++20 -O2 -I. -DUSE_CUDA benchmark_matmul.cpp gpt/Tensor/Tensor.cpp gpt/Tensor/TensorCuda.cu -o bench_cuda
//
#include "gpt/Tensor/Tensor.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

struct Case { const char* label; size_t batch, M, K, N; int reps; };

int main()
{
#ifdef USE_CUDA
    std::printf("Build: CUDA (GPU path active above threshold)\n\n");
#else
    std::printf("Build: CPU only\n\n");
#endif

    std::vector<Case> cases = {
        // Mirrors this project's default demo config: embedDim=16, seqLen=16,
        // 1 head -> attention scores are (1,16,16)x(1,16,16). All matmuls
        // in that model are this size or smaller.
        {"default demo config: attn scores (1x16x16 @ 1x16x16)",      1,  16, 16, 16,  20000},
        // A "real" small GPT: embedDim=256, seqLen=128, 8 heads (headDim=32)
        {"small-GPT attn scores (8x128x32 @ 8x32x128)",                8, 128, 32, 128, 500},
        {"small-GPT attn@V      (8x128x128 @ 8x128x32)",               8, 128, 128, 32, 500},
        // LM head-sized: embedDim=256 -> vocab=5000, seqLen=128 (as a single big matmul)
        {"LM-head-sized (1x128x256 @ 1x256x5000)",                     1, 128, 256, 5000, 20},
    };

    for (auto& c : cases)
    {
        Tensor A({c.batch, c.M, c.K});
        Tensor B({c.batch, c.K, c.N});
        A.random(0.0, 0.02);
        B.random(0.0, 0.02);

        // warmup
        Tensor::matmul(A, B);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < c.reps; ++i)
        {
            Tensor C = Tensor::matmul(A, B);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_call_ms = total_ms / c.reps;
        double flops = 2.0 * c.batch * c.M * c.K * c.N; // multiply+add per element
        double gflops_per_sec = (flops / (per_call_ms / 1000.0)) / 1e9;

        std::printf("%-55s reps=%-6d  %8.4f ms/call   %8.3f GFLOP/s\n",
                    c.label, c.reps, per_call_ms, gflops_per_sec);
    }
    return 0;
}
