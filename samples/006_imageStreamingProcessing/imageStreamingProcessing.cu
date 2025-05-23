#include <opencv2/opencv.hpp>
#include <iostream>
#include "../utils/utils.hpp"
__global__ void grayscaleKernel(cudaTextureObject_t texObj, unsigned char* out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    
    uchar4 pixel = tex2D<uchar4>(texObj, x, y);
    out[y * width + x] = (pixel.x + pixel.y + pixel.z) / 3;
}

class CudaImageResource {
    public: 
        int width;
        int height;
        int step;
        cudaArray* cuArray = nullptr;
        cudaTextureObject_t texObj = 0;
        unsigned char* d_input = nullptr;
        unsigned char* d_output = nullptr;
        unsigned char* h_input_pinned = nullptr;
        unsigned char* h_output_pinned = nullptr;
        cudaStream_t stream = nullptr;
        cv::Mat source;
        cv::Mat output;
        dim3 block;
        dim3 grid;
        void Destroy() {
            if (texObj != 0) cudaDestroyTextureObject(texObj);
            if (cuArray != nullptr) cudaFreeArray(cuArray);
            if (d_output != nullptr) cudaFree(d_output);
            if (h_input_pinned != nullptr) cudaFreeHost(h_input_pinned);
            if (h_output_pinned != nullptr) cudaFreeHost(h_output_pinned);
            if (stream != nullptr) cudaStreamDestroy(stream);
        }
};


__global__ void grayscaleKernel_NoTexture(unsigned char* input, unsigned char* output, 
                                        int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4; // BGRA format
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    output[y * width + x] = (r + g + b) / 3;
}

void use_StreamProcess_NoTexture(ImageStocker stocker)
{
    std::vector<CudaImageResource> resources;
    int N = stocker.NumImages();
    int numStreams = std::min(N, 4);
    resources.resize(N);
    
    std::vector<cudaStream_t> streams(2);
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Initialize all resources
    for (int i = 0; i < N; i++) {
        resources[i].stream = streams[i % 2];
        
        cv::cvtColor(stocker.Get(i), resources[i].source, cv::COLOR_BGR2BGRA);
        resources[i].width = resources[i].source.cols;
        resources[i].height = resources[i].source.rows;
        
        size_t input_size = resources[i].width * resources[i].height * 4;
        size_t output_size = resources[i].width * resources[i].height;
        
        // Pinned memory allocation
        cudaMallocHost((void**)&resources[i].h_input_pinned, input_size);
        cudaMallocHost((void**)&resources[i].h_output_pinned, output_size);
        cudaMalloc(&resources[i].d_input, input_size);
        cudaMalloc(&resources[i].d_output, output_size);
        
        memcpy(resources[i].h_input_pinned, resources[i].source.data, input_size);
        resources[i].output = cv::Mat(resources[i].height, resources[i].width, 
                                    CV_8UC1, resources[i].h_output_pinned);
        
        resources[i].block = dim3(16, 16);
        resources[i].grid = dim3((resources[i].width + 15) / 16, 
                               (resources[i].height + 15) / 16);
    }

    // Pipeline execution with smaller batches
    int batchSize = 2; 
    for (int batch = 0; batch < N; batch += batchSize) {
        int endBatch = std::min(batch + batchSize, N);
        
        // H2D transfers for this batch
        for (int i = batch; i < endBatch; i++) {
            cudaMemcpyAsync(resources[i].d_input, resources[i].h_input_pinned,
                           resources[i].width * resources[i].height * 4,
                           cudaMemcpyHostToDevice, resources[i].stream);
        }
        
        // Kernel executions for this batch
        for (int i = batch; i < endBatch; i++) {
            grayscaleKernel_NoTexture<<<resources[i].grid, resources[i].block, 0, resources[i].stream>>>
                (resources[i].d_input, resources[i].d_output, resources[i].width, resources[i].height);
        }
        
        // D2H transfers for this batch
        for (int i = batch; i < endBatch; i++) {
            cudaMemcpyAsync(resources[i].h_output_pinned, resources[i].d_output,
                           resources[i].width * resources[i].height,
                           cudaMemcpyDeviceToHost, resources[i].stream);
        }
        
        // Synchronize this batch before moving to next
        for (int i = batch; i < endBatch; i++) {
            cudaStreamSynchronize(resources[i].stream);
        }
    }
    
    // Save results and cleanup
    for (int i = 0; i < N; i++) {
        std::string outPath = "006_cuda-stream-notexobj_img_" + ZeroPadding(2, i) + ".png";
        cv::imwrite(outPath, resources[i].output);
        resources[i].Destroy();
    }
}

void use_StreamProcess(ImageStocker stocker)
{
    // Create Stream
    std::vector<CudaImageResource> resources;
    int N = stocker.NumImages();
    resources.resize(N);

    std::vector<cudaStream_t> streams(2);
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);    

    // Initialize
    for (int i = 0; i < N; i++)
    {
        resources[i].stream = streams[i % 2];

        // cudaArray
        cv::cvtColor(stocker.Get(i), resources[i].source, cv::COLOR_BGR2BGRA);
        resources[i].width = resources[i].source.cols;
        resources[i].height = resources[i].source.rows;
        resources[i].step = resources[i].source.step;

        size_t input_size = resources[i].width * resources[i].height * 4 * sizeof(unsigned char);
        size_t output_size = resources[i].width * resources[i].height * sizeof(unsigned char);

        cudaMallocHost((void**)&resources[i].h_input_pinned, input_size);
        cudaMallocHost((void**)&resources[i].h_output_pinned, output_size);

        memcpy(resources[i].h_input_pinned, resources[i].source.ptr<uchar4>(),
            resources[i].step * resources[i].height);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaMallocArray(&resources[i].cuArray, &channelDesc, resources[i].width, resources[i].height);

        // Texture Object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = resources[i].cuArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaCreateTextureObject(&resources[i].texObj, &resDesc, &texDesc, nullptr);

        cudaMalloc(&resources[i].d_output, resources[i].width * resources[i].height * sizeof(unsigned char));

        // Kernel call
        resources[i].block = dim3(16, 16);
        resources[i].grid = dim3((resources[i].width + 15) / 16, (resources[i].height + 15) / 16);

        // Save Image
        resources[i].output = cv::Mat(resources[i].height, resources[i].width, CV_8UC1);
    }

    // Set Tasks
    for (int i = 0; i < N; i++)
    {
        cudaMemcpy2DToArrayAsync(resources[i].cuArray, // target
            0, 0, resources[i].h_input_pinned,
            resources[i].step, 
            resources[i].width * sizeof(uchar4), 
            resources[i].height, // image 
            cudaMemcpyHostToDevice, 
            resources[i].stream); // direction
    }

        // Pipeline execution with smaller batches
    int batchSize = 2; 
    for (int batch = 0; batch < N; batch += batchSize) {
        int endBatch = std::min(batch + batchSize, N);
        
        for (int i = batch; i < endBatch; i++)
        {
            grayscaleKernel<<<resources[i].grid, resources[i].block, 0, resources[i].stream>>>
                (resources[i].texObj, resources[i].d_output, resources[i].width, resources[i].height);
        }

        for (int i = batch; i < endBatch; i++)
        {
            cudaMemcpyAsync(resources[i].h_output_pinned, resources[i].d_output, 
                resources[i].width * resources[i].height * sizeof(unsigned char),
                cudaMemcpyDeviceToHost, resources[i].stream);
        }

        for (int i = batch; i < endBatch; i++)
        {
            cudaStreamSynchronize(resources[i].stream);
        }
    }

    for (int i = 0; i < N; i++)
    {
        memcpy(resources[i].output.data, resources[i].h_output_pinned, resources[i].width * resources[i].height * sizeof(unsigned char));
        std::string outPath = "006_cuda-stream_img_" + ZeroPadding(2, i) + ".png";
        if (!cv::imwrite(outPath, resources[i].output))
        {
            std::cerr << "Failed to save iamge to " << outPath << std::endl;
            return ;
        }
        std::cout << "Saved image to: " << outPath << std::endl;
        resources[i].Destroy();
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir>" << std::endl;
        return 1;
    }

    std::string imgDir = argv[1];
    ImageStocker stocker;
    stocker.LoadImages(imgDir);
    use_StreamProcess(stocker);
    use_StreamProcess_NoTexture(stocker);
    return 0;
}