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

void processSingleImage(ImageStocker& stocker, int index, cudaStream_t stream)
{

}

class CudaImageResource {
    public: 
        int width;
        int height;
        cudaArray* cuArray = nullptr;
        cudaTextureObject_t texObj = 0;
        unsigned char* d_output = nullptr;
        cudaStream_t stream = nullptr;
        cv::Mat source;
        cv::Mat output;
        dim3 block;
        dim3 grid;
        void Destroy() {
            if (texObj != 0) cudaDestroyTextureObject(texObj);
            if (cuArray != nullptr) cudaFreeArray(cuArray);
            if (d_output != nullptr) cudaFree(d_output);
            if (stream != nullptr) cudaStreamDestroy(stream);
        }
};

void use_StreamProcess(ImageStocker stocker)
{
    // Create Stream
    std::vector<CudaImageResource> resources;
    int N = stocker.NumImages();
    resources.resize(N);
    for (int i = 0; i < N; i++)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        resources[i].stream = stream;
    }

    // Initialize
    for (int i = 0; i < N; i++)
    {
        // cudaArray
        cv::cvtColor(stocker.Get(i), resources[i].source, cv::COLOR_BGR2BGRA);
        resources[i].width = resources[i].source.cols;
        resources[i].height = resources[i].source.rows;
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
        resources[i].block = dim3(4, 4);
        resources[i].grid = dim3((resources[i].width + 3) / 4, (resources[i].height + 3) / 4);

        // Save Image
        resources[i].output = cv::Mat(resources[i].height, resources[i].width, CV_8UC1);
    }

    // Set Tasks
    for (int i = 0; i < N; i++)
    {
        cudaMemcpy2DToArrayAsync(resources[i].cuArray, // target
            0, 0, resources[i].source.ptr<uchar4>(), resources[i].source.step, resources[i].width * sizeof(uchar4), resources[i].height, // image 
            cudaMemcpyHostToDevice, resources[i].stream); // direction
    }

        for (int i = 0; i < N; i++)
    {
        grayscaleKernel<<<resources[i].grid, resources[i].block, 0, resources[i].stream>>>(resources[i].texObj, resources[i].d_output, resources[i].width, resources[i].height);
    }
        for (int i = 0; i < N; i++)
    {
        cudaMemcpyAsync(resources[i].output.data, resources[i].d_output, resources[i].width * resources[i].height * sizeof(unsigned char), cudaMemcpyDeviceToHost, resources[i].stream);
    }

    for (int i = 0; i < N; i++)
    {
        cudaStreamSynchronize(resources[i].stream);
    }

    for (int i = 0; i < N; i++)
    {
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
    return -1;
}