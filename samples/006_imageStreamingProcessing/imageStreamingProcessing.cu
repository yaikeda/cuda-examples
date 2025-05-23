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

void processSingleImage(ImageStocker& stocker, int index)
{
    // cudaArray
    cv::Mat imageBGRA;
    cv::cvtColor(stocker.Get(index), imageBGRA, cv::COLOR_BGR2BGRA);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, imageBGRA.cols, imageBGRA.rows);
    cudaMemcpy2DToArray(cuArray, // target
        0, 0, imageBGRA.ptr<char4>(), imageBGRA.step, imageBGRA.cols * sizeof(uchar4), imageBGRA.rows, // image 
        cudaMemcpyHostToDevice); // direction

    // Texture Object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    int width = imageBGRA.cols;
    int height = imageBGRA.rows;
    unsigned char* d_output;
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    // Kernel call
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<grid, block>>>(texObj, d_output, width, height);

    // Save Image
    cv::Mat outputGray(height, width, CV_8UC1);
    cudaMemcpy(outputGray.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    std::string outPath = "006_cuda-stream_img_" + ZeroPadding(2, index) + ".png";
    if (!cv::imwrite(outPath, outputGray))
    {
        std::cerr << "Failed to save iamge to " << outPath << std::endl;
        return ;
    }
    std::cout << "Saved image to: " << outPath << std::endl;

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
}

void use_StreamProcess(ImageStocker stocker)
{
    // Create Stream
    std::vector<cudaStream_t> streams;
    int N = stocker.NumImages();
    streams.resize(N);
    for (int i = 0; i < N; i++)
    {
        cudaStreamCreate(&streams[i]);
        processSingleImage(stocker, i);
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