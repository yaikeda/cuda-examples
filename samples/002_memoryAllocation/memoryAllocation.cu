#include <cuda_runtime.h>
#include <iostream>

const size_t SIZE = 1L << 30;

bool isSuccess(cudaError_t err) {
    if (err != cudaSuccess)
    {
        std::cerr << "cuda memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

void* use_cudaMalloc() {
    void* device_ptr = nullptr;
    
    if (isSuccess(cudaMalloc(&device_ptr, SIZE)))
    {
        std::cout << "cudaMalloc succeeded. Pointer = " << device_ptr << std::endl;
        return device_ptr;
    }
    return nullptr;
}

void* use_cudaMallocManaged() {
    void* unified_ptr = nullptr;
    
    if (isSuccess(cudaMallocManaged(&unified_ptr, SIZE)))
    {
        std::cout << "cudaMallocManaged succeeded. Pointer = " << unified_ptr << std::endl;
        return unified_ptr;
    }
    return nullptr;
}

void* use_cudaHostAlloc() {
    void* host_ptr = nullptr;
    if(isSuccess(cudaHostAlloc(&host_ptr, SIZE, cudaHostAllocDefault)))
    {
        std::cout << "cudaHostAlloc succeeded. Pointer = " << host_ptr << std::endl;
        return host_ptr;
    }
    return nullptr;
}

int main() 
{
    void* device_ptr = use_cudaMalloc();
    void* unified_ptr = use_cudaMallocManaged();
    void* host_ptr = use_cudaHostAlloc();

    std::cin.get(); // pause

    cudaFree(device_ptr);
    cudaFree(unified_ptr);
    cudaFree(host_ptr);
}