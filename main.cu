#include <iostream>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return -1;
    }

    std::cout << "CUDA Devices Found: "
              << deviceCount
              << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: "
              << prop.major << "."
              << prop.minor << std::endl;

    helloKernel<<<1, 8>>>();

    cudaDeviceSynchronize();

    return 0;
}