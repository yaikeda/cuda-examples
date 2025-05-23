#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

// Zero fill function
std::string ZeroPadding(int digits, int num)
{
    std::ostringstream oss;
    oss << std::setw(digits) << std::setfill('0') << num;
    return oss.str();
}

std::vector<std::string> GetImageFiles(const std::string& dirPath)
{
    std::vector<std::string> result;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath))
    {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        auto ext = path.extension().string();
        if (ext == ".jpg" || ext == ".png" || ext == ".bmp")
        {
            result.push_back(path.string());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// Small Image Load and Stock class
class ImageStocker
{
    public:
        bool LoadImages(const std::string& dirPath)
        {
            auto files = GetImageFiles(dirPath);
            for (const std::string& file : files)
            {
                std::cout << "Image Path: " << file << std::endl;
                cv::Mat img = cv::imread(file);
                if (img.empty()) {
                    printf("failed to load image\n");
                    return false;
                }
                m_images.push_back(img);
            }
            return true;
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