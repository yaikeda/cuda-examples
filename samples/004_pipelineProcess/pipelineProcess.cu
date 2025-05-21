#include <opencv2/opencv.hpp>
#include <iostream>

// Expected input: a.exe C:\path\to\image\folder num-images
int main(int argc, char** argv) {
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir> <num_images>" << std::endl;
        return 1;
    }
    std::string imgDir = argv[1];
    size_t imgNum = std::atoi(argv[2]);
    std::cout << "Target Dir: " << imgDir << " Num images: " << imgNum << std::endl;

    std::string imgPath = imgDir + "/img_01.png";
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        printf("failed to load image\n");
        return -1;
    }
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}