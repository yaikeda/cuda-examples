#include <opencv2/opencv.hpp>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <texture_types.h>

__global__ void grayscaleKernel(cudaTextureObject_t texObj, unsigned char* out, int width, int height) // New API can take texObj directly!!!
{
    // this is 2D then blockIdx also 2D.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 pixel = tex2D<uchar4>(texObj, x, y);
    unsigned char gray = (pixel.x * pixel.y + pixel.z) / 3;
    out[y * width + x] = gray;
}

// Zero fill function
std::string ZeroPadding(int digits, int num)
{
    std::ostringstream oss;
    oss << std::setw(digits) << std::setfill('0') << num;
    return oss.str();
}

// Small Image Load and Stock class
class ImageStocker
{
    public:
        ImageStocker(std::string dir, int count)
        {
            for (int i = 1; i <= count; i++)
            {
                std::string imgPath = dir + "/img_" + ZeroPadding(2, i) + ".png";
                std::cout << "Image Path: " << imgPath << std::endl;
                cv::Mat img = cv::imread(imgPath);
                if (img.empty()) {
                    printf("failed to load image\n");
                    break;
                }
                m_images.push_back(img);
            }
        } 
        int NumImages()
        {
            return m_images.size();
        }
        cv::Mat Get(int id)
        {
            if (NumImages() <= id)
            {
                std::cout << "m_images.size() <= id" << std::endl;
                return cv::Mat();
            }
            return m_images[id];
        }
    private: 
        std::vector<cv::Mat> m_images;
};

void use_OpenCV(ImageStocker stocker)
{
    for (int i = 0; i < stocker.NumImages(); i++)
    {
        cv::Mat grayImage;
        cv::cvtColor(stocker.Get(i), grayImage, cv::COLOR_BGR2GRAY);
        std::string outpath = "004_opencv_img_" + ZeroPadding(2, i) + ".png";
        if (!cv::imwrite(outpath, grayImage)) {
            std::cerr << "Failed to save image to " << outpath << std::endl;
            continue;
        }
        std::cout << "Saved image to: " << outpath << std::endl;
    }
}

void use_SequentialProcess(ImageStocker stocker)
{
    for (int i = 0; i < stocker.NumImages(); i++)
    {
        // Set Image
        cv::Mat imageBGRA;
        cv::cvtColor(stocker.Get(i), imageBGRA, cv::COLOR_BGR2BGRA);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &channelDesc, imageBGRA.cols, imageBGRA.rows); // cudaMallocArray needs image information (like ndarray to Unity Texture2D, row/col/depth info)
        cudaMemcpy2DToArray(cuArray, 0, 0, // Memcpy to cache. It looks like BufferObject in OpenGL 
            imageBGRA.ptr<uchar4>(), imageBGRA.step,
            imageBGRA.cols * sizeof(uchar4), imageBGRA.rows,
            cudaMemcpyHostToDevice);
        
        // Setting Descs
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Texture Object
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

        // Device Memory
        int width = imageBGRA.cols;
        int height = imageBGRA.rows;
        unsigned char* d_output;
        cudaMalloc(&d_output, width * height * sizeof(unsigned char));

        // Prepare for kernel calling
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        grayscaleKernel<<<grid, block>>>(texObj, d_output, width, height);


        // Save Image
        cv::Mat outputGray(height, width, CV_8UC1);
        cudaMemcpy(outputGray.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        std::string outpath = "004_cuda-sequential_img_" + ZeroPadding(2, i) + ".png";
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

// Expected input: a.exe C:\path\to\image\folder num-images
int main(int argc, char** argv) {
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir> <num_images>" << std::endl;
        return 1;
    }
    std::string imgDir = argv[1];
    size_t imgNum = std::atoi(argv[2]);
    ImageStocker stocker(imgDir, imgNum);
    use_OpenCV(stocker);
    use_SequentialProcess(stocker);

    // Check images with GUI
    // for (int i = 0; i < stocker.NumImages(); i++)
    // {
    //     cv::imshow("Test", stocker.Get(i));
    //     cv::waitKey(0);
    // }
    return 0;
}