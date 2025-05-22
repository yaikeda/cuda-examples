#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

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