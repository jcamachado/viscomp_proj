#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

#define IMAGE_DIR "images/q3"

int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    // Load the image in grayscale
    cv::Mat src = cv::imread((image_dir / "Questionario-3-Imagem-3.tif").string(), cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "Cannot open image file." << std::endl;
        return -1;
    }

    // Apply mean filter to remove salt and pepper noise
    cv::Mat denoised;
    cv::blur(src, denoised, cv::Size(3, 3));
    cv::imwrite((image_dir / "q3_im3_denoised3x3.png").string(), denoised);

    cv::blur(src, denoised, cv::Size(5, 5));
    cv::imwrite((image_dir / "q3_im3_denoised5x5.png").string(), denoised);
    
    cv::blur(src, denoised, cv::Size(7, 7));
    cv::imwrite((image_dir / "q3_im3_denoised7x7.png").string(), denoised);

    return 0;
}