#include <opencv2/opencv.hpp>
#include <iostream>

#include <cuda_runtime.h>

#include "../utils/utils.hpp"

__global__ void grayscaleKernel(cudaTextureObject_t texObj, unsigned char* out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 pixel = tex2D<uchar4>(texObj, x, y);
    unsigned char gray = (pixel.x + pixel.y + pixel.z) / 3;
    out[y * width + x] = gray;
}

void use_SequentialProcess(ImageStocker stocker)
{
    for (int i = 0; i < stocker.NumImages(); i++)
    {
        cv::Mat imageBGRA;
        cv::cvtColor(stocker.Get(i), imageBGRA, cv::COLOR_BGR2BGRA);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &channelDesc, imageBGRA.cols, imageBGRA.rows);
        cudaMemcpy2DToArray(cuArray, 0, 0,
            imageBGRA.ptr<uchar4>(), imageBGRA.step,
            imageBGRA.cols * sizeof(uchar4), imageBGRA.rows,
            cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // TextureObject
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

        // Device Memory Alloc
        int width = imageBGRA.cols;
        int height = imageBGRA.rows;
        unsigned char* d_output;
        cudaMalloc(&d_output, width * height * sizeof(unsigned char));

        // prepare for kernel call
        dim3 block(16, 16);
        dim3 grid((width +15) / 16, (height + 15) / 16);
        grayscaleKernel<<<grid, block>>>(texObj, d_output, width, height);

        // Save Image
        cv::Mat outputGray(height, width, CV_8UC1);
        cudaMemcpy(outputGray.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        std::string outpath = "005_cuda-sequential_debug_img_" + ZeroPadding(2, i) + ".png";
        if (!cv::imwrite(outpath, outputGray)) {
            std::cerr << "Failed to save image to " << outpath << std::endl;
            return;
        }
        std::cout << "Saved image to: " << outpath << std::endl;

        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        cudaFree(d_output);
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir>" << std::endl;
        return 1;
    }
    std::string imgDir = argv[1];
    ImageStocker stocker;
    stocker.LoadImages(imgDir);
    use_SequentialProcess(stocker);
    return 0;
}

