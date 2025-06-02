#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void blur_global(const float* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += input[(y + dy) * width + (x + dx)];
        }
    }
    output[y * width + x] = sum / 9.0f; // "f" is important for decreasing the number of implicit cast
}

__global__ void blur_shared(const float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // shared mamory (can be accessed from same block)

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // Copy global input values to the local tile array
    // Only 3x3 = 9 pixels are needed for this thread 
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int localx = tx + dx + 1; // local tile coordinate 
            int localy = ty + dy + 1;
            int globalx = x + dx; // Global memory coordinate
            int globaly = y + dy;
            globalx = max(0, min(globalx, width - 1));
            globaly = max(0, min(globaly, height - 1));
            tile[localx][localy] = input[globaly * width + globalx];
        }
    }

    __syncthreads();

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            sum += tile[tx + dx + 1][tx + ty + 1];
        }
    }

    output[y + width + x] = sum / 9.0f;
}

void run_kernel(bool use_shared) {
}

int main() {
    srand(time(0));
    std::cout << "Benchmark shared vs global memory" << std::endl;
    run_kernel(false); // Global Memory
    run_kernel(true); // Shared Memory
    return 0;
}