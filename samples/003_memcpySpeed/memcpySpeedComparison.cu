#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

constexpr size_t SIZE = 2ULL << 30;
const size_t NUM_BLOCKS = 512;
const size_t NUM_THREADS = 128;

__global__ void doNothingCudaKernel(char* data, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        volatile char tmp = data[idx];
    }
}

bool isSuccess(cudaError_t err) {
    if (err != cudaSuccess)
    {
        std::cerr << "cuda memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

class AutoTimeLogger
{
    public:
        AutoTimeLogger(std::string label)
        {
            m_label = label;
            m_start = std::chrono::high_resolution_clock::now();
        }
        ~AutoTimeLogger()
        {
            m_end = std::chrono::high_resolution_clock::now();
            showLog();
        }
    private: 
        std::string m_label;
        std::chrono::steady_clock::time_point m_start;
        std::chrono::steady_clock::time_point m_end;
        void showLog()
        {
            double elapsed = std::chrono::duration<double, std::milli>(m_end - m_start).count();
            std::cout << m_label << " Elapsed time: " << elapsed << " ms" << std::endl;
        }
};

// UVA
void use_cudaMalloc() {
    void* device_ptr = nullptr;
    if (isSuccess(cudaMalloc(&device_ptr, SIZE)))
    {
        void* host_ptr = malloc(SIZE);
        memset(host_ptr, 111, SIZE); // I like 111

        {
            AutoTimeLogger t("cudaMalloc HtoD");
            cudaMemcpy(device_ptr, host_ptr, SIZE, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }

        {
            AutoTimeLogger t("cudaMalloc DtoH");
            cudaMemcpy(host_ptr, device_ptr, SIZE, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        free(host_ptr);
        cudaFree(device_ptr);
    }
}

// UVM
void use_cudaMallocManaged() { 
    char* unified_ptr = nullptr;
    if (isSuccess(cudaMallocManaged(&unified_ptr, SIZE)))
    {
        memset(unified_ptr, 111, SIZE); // Write data on cpu
        {
            int device = 0;
            cudaGetDevice(&device);
            AutoTimeLogger t("cudaMallocManaged HtoD PrefetchAsync");
            cudaMemPrefetchAsync(unified_ptr, SIZE, device);
            cudaDeviceSynchronize();
        }
        {
            AutoTimeLogger t("cudaMallocManaged HtoD Call kernel");
            doNothingCudaKernel<<<NUM_BLOCKS, NUM_THREADS>>>(unified_ptr, SIZE);
            cudaDeviceSynchronize();
        }
        {
            doNothingCudaKernel<<<NUM_BLOCKS, NUM_THREADS>>>(unified_ptr, SIZE);
            cudaDeviceSynchronize();
            AutoTimeLogger t("cudaMallocManaged DtoH ");
            std::cerr << "Refer unified_ptr from CPU: " << unified_ptr[0] << std::endl;
        }
    }
}

// Zero-copy
void use_cudaHostAllocMapped() {
    char* host_ptr = nullptr;
    if (isSuccess(cudaHostAlloc((void**)&host_ptr, SIZE, cudaHostAllocMapped)))
    {
        char* device_ptr = nullptr;
        cudaHostGetDevicePointer((void**)&device_ptr, host_ptr, 0);

        {
            AutoTimeLogger t("Zero-Copy HtoD");
            doNothingCudaKernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_ptr, SIZE);
            cudaDeviceSynchronize();
        }
        {
            AutoTimeLogger t("Zero-Copy DtoH");
            std::cout << "host_ptr[0] = " << (int)host_ptr[0] << std::endl;
        }
    }
}

int main()
{
    use_cudaMalloc();
    use_cudaMallocManaged();
    use_cudaHostAllocMapped();
    return 0;
}