#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16
#define REPEAT 1000
#define WIDTH 1000
#define HEIGHT 1000
#define FILTER_SIZE 3 // (FILTER_SIZE - 1) * 0.5  (ex: 3->1, 5->2) 

__global__ void blur_global(const float* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < FILTER_SIZE || x >= width - FILTER_SIZE || y < FILTER_SIZE || y >= height - FILTER_SIZE) return;

    float sum = 0.0f;
    for (int dy = -FILTER_SIZE; dy <= FILTER_SIZE; dy++) {
        for (int dx = -FILTER_SIZE; dx <= FILTER_SIZE; dx++) {
            sum += input[(y + dy) * width + (x + dx)];
        }
    }
    output[y * width + x] = sum / (FILTER_SIZE * 2 + 1); // "f" is important for decreasing the number of implicit cast
}

__global__ void blur_shared(const float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE + FILTER_SIZE * 2][BLOCK_SIZE + FILTER_SIZE * 2]; // shared mamory (can be accessed from same block)

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    int globalx = globalx = max(0, min(x, width - FILTER_SIZE));
    int globaly = globaly = max(0, min(y, height - FILTER_SIZE));
    tile[tx + 1][ty + 1] = input[globaly * width + globalx]; // Copy value 1 thread 1 global access

    __syncthreads();

    if (x < FILTER_SIZE || x >= width - FILTER_SIZE || y < FILTER_SIZE || y >= height - FILTER_SIZE) return;

    float sum = 0.0f;
    for (int dy = -FILTER_SIZE; dy <= FILTER_SIZE; dy++)
    {
        for (int dx = -FILTER_SIZE; dx <= FILTER_SIZE; dx++)
        {
            sum += tile[tx + dx + FILTER_SIZE][tx + ty + FILTER_SIZE];
        }
    }

    output[y + width + x] = sum / (FILTER_SIZE * 2 + 1);
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

    std::cout << (use_shared? "[shared] " : "[global] ") << "REPEAT "<< REPEAT << " FILTER " << FILTER_SIZE << " Time : " << milliseconds << " ms" << std::endl;

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