#include <cuda_runtime.h>
#include <iostream>

const size_t SIZE = 1L << 30;

bool isSuccess(cudaError_t err) {
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

void* use_cudaMalloc() {
    void* device_ptr = nullptr;
    size_t size = SIZE; 

    if (isSuccess(cudaMalloc(&device_ptr, size)))
    {
        std::cout << "cudaMalloc succeeded. Pointer = " << device_ptr << std::endl;
        return device_ptr;
    }
    return nullptr;
}

void* use_cudaMallocManaged() {
    void* unified_ptr = nullptr;
    size_t size = SIZE;

    if (isSuccess(cudaMallocManaged(&unified_ptr, size)))
    {
        std::cout << "cudaMallocManaged succeeded. Pointer = " << unified_ptr << std::endl;
        return unified_ptr;
    }
    return nullptr;
}

int main() 
{
    void* device_ptr = use_cudaMalloc();
    void* unified_ptr = use_cudaMallocManaged();
    
    std::cin.get(); // pause

    cudaFree(device_ptr);
    cudaFree(unified_ptr);
}