#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Device 
__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    // Reference: https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf
    int i = // array index
        threadIdx.x + // Thread ID in current Block
        blockIdx.x * // Block ID in current Grid
        blockDim.x; // Number of Threads in current Block
    if (i < n) // array index out of scope check
    {
        c[i] = a[i] + b[i];
    }
}

// CPU
void vectorAddCPU(const float* A, const float* B, float* c, const int N)
{
    for (int i = 0; i < N; i++)
    {
        c[i] = A[i] + B[i];
    }
}

int main() {
    //const int N = 512; // CPU win
    //const int N = 1000000; // 1M GPU win
    const int N =   1000000000; // 1G GPU win
    size_t size = N * sizeof(float); // calc allocation size

    // --- CUDA ---

    // host
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i*2);
    }

    // device
    float *d_a, *d_b, *d_c; // VRAM address
    cudaMalloc(&d_a, size); // allocate CUDA memory for N float values
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // memo: cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    // cudaMemcpyKind? CUDA memory copy types
    // - 0: h -> h == cudaMemcpyHostToHost
    // - 1: h -> d == cudaMemcpyHostToDevice 
    // - 2: d -> h == cudaMemcpyDeviceToHost
    // - 3: d -> d == cudaMemcpyDeviceToDevice // multiple GPU environment would use this
    // - 4: auto
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); // size is in bytes
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    // Note: manual version of ceil 
    // if N % threadsPerBlock == 0, number of blocksPerGrid not enough
    // ex) N is 1000, 1000 / 256 is 3 (in int). 
    // 3 * threadsPerBlock = 768
    // some vectors will not be calculated.
    // to include all N, we need to take ceil here
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // get result
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;
    std::cout << h_c[10] << " " << h_c[100] << " " << h_c[255] << std::endl;
    // --- CUDA END ---

    // --- CPU ---
    for (int i = 0; i < N; i++) h_c[i] = 0;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << h_c[10] << " " << h_c[100] << " " << h_c[255] << std::endl;
    // --- CPU END ---

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}

