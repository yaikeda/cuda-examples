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

void* use_cudaHostAlloc_cudaHostAllocMapped()
{
    void* h_ptr = nullptr;

    if(isSuccess(cudaHostAlloc(&h_ptr, SIZE, cudaHostAllocMapped)))
    {
        void* d_ptr = nullptr;
        if(isSuccess(cudaHostGetDevicePointer(&d_ptr, h_ptr, 0))) // flag must be zero for now
        {
            std::cout << "Zero-copy mapping succeeded. Host = " << h_ptr << ", Device =" << d_ptr << std::endl;
            return h_ptr;
        }
        cudaFree(h_ptr);
        return nullptr;
    }
    return nullptr;
}

int main() 
{
    void* device_ptr = use_cudaMalloc();
    void* unified_ptr = use_cudaMallocManaged();
    void* host_ptr = use_cudaHostAlloc();
    void* host_ptr2 = use_cudaHostAlloc_cudaHostAllocMapped();

    std::cin.get(); // pause

    cudaFree(device_ptr);
    cudaFree(unified_ptr);
    cudaFree(host_ptr);
    cudaFree(host_ptr2);
}