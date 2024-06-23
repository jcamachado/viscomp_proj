#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

#define IMAGE_DIR "images/q3"

void eqHist(const cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    int histSize = 256; // Number of bins for a 8-bit image
    std::vector<int> histogram(histSize, 0);
    std::vector<float> cdf(histSize, 0);    // Cumulative Distribution Function

    // Calculate the histogram
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            histogram[src.at<uchar>(y, x)]++;
        }
    }

    // Calculate the CDF
    cdf[0] = histogram[0];
    for (int i = 1; i < histSize; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Normalize the CDF
    float total = src.rows * src.cols; // w * h
    for (int i = 0; i < histSize; i++) {    // For each intensity level 
        cdf[i] /= total;                    // Divide by the total number of pixels
    }

    // Map back the intensity levels
    // g(x, y) = K * cdf[f(x, y)] K is the maximum intensity level (255)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // g(x, y) =          K   * c  [f(x, y)]
            dst.at<uchar>(y, x) = 255 * cdf[src.at<uchar>(y, x)];
        }
    }
    
}

int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    // Load the image in grayscale
    cv::Mat src = cv::imread((image_dir / "Questionario-3-Imagem-1.tif").string(), cv::IMREAD_GRAYSCALE);
    cv::Mat src2 = cv::imread((image_dir / "Questionario-3-Imagem-2.tif").string(), cv::IMREAD_GRAYSCALE);
    if (src.empty() || src2.empty()) {
        std::cout << "Cannot open image file." << std::endl;
        return -1;
    }

    // Equalize the histogram of the grayscale image
    cv::Mat equalized, equalized2;
    eqHist(src, equalized);
    eqHist(src2, equalized2);

    // Save the equalized image
    cv::imwrite((image_dir / "q3_im1_equalized.png").string(), equalized);
    cv::imwrite((image_dir / "q3_im2_equalized.png").string(), equalized2);

    return 0;
}