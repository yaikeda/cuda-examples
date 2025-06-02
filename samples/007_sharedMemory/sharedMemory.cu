#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16
#define REPEAT 1000
#define WIDTH 1000
#define HEIGHT 1000

__global__ void blur_global(const float* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

    float sum = 0.0f;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            sum += input[(y + dy) * width + (x + dx)];
        }
    }
    sum = sqrtf(sum * sum + 0.001f);
    output[y * width + x] = sum / 9.0f; // "f" is important for decreasing the number of implicit cast
}

__global__ void blur_shared(const float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4]; // shared mamory (can be accessed from same block)

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // Copy global input values to the local tile array
    // Only 3x3 = 9 pixels are needed for this thread 
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int localx = tx + dx + 2; // local tile coordinate 
            int localy = ty + dy + 2;
            int globalx = x + dx; // Global memory coordinate
            int globaly = y + dy;
            globalx = max(0, min(globalx, width - 2));
            globaly = max(0, min(globaly, height - 2));
            tile[localx][localy] = input[globaly * width + globalx];
        }
    }

    __syncthreads();

    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

    float sum = 0.0f;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            sum += tile[tx + dx + 2][tx + ty + 2];
        }
    }

    sum = sqrtf(sum * sum + 0.001f);
    output[y + width + x] = sum / 9.0f;
}

void run_kernel(bool use_shared) {
    float* d_input;
    float* d_output;
    size_t size = WIDTH * HEIGHT * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Init
    float* h_input = new float[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = rand() % 256;
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < REPEAT; i++) {
        if (use_shared)
            blur_shared<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
        else 
            blur_global<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << (use_shared? "[shared] " : "[global] ") << "Time (" << REPEAT << ") : " << milliseconds << " ms" << std::endl;

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    srand(time(0));
    std::cout << "Benchmark shared vs global memory" << std::endl;
    run_kernel(false); // Global Memory
    run_kernel(true); // Shared Memory
    return 0;
}