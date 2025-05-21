#include <opencv2/opencv.hpp>
#include <iostream>

std::string ZeroPadding(int digits, int num)
{
    std::ostringstream oss;
    oss << std::setw(digits) << std::setfill('0') << num;
    return oss.str();
}

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
            std::cout << NumImages() << " " << id << std::endl;
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
    
    for (int i = 0; i < stocker.NumImages(); i++)
    {
        cv::imshow("Test", stocker.Get(i));
        cv::waitKey(0);
    }
    return 0;
}