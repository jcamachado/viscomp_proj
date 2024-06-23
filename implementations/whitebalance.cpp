#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <filesystem>

#define IMAGE_DIR "images/"

void performWhiteBalance(const cv::Mat& src, cv::Mat& dst) {
    // cv::xphoto::createSimpleWB()->balanceWhite(src, dst); // Default from OpenCV
    /*
        Calculate the average color of the whole image
        mean reaturns a Scalar object with 4 elements: B, G, R, A
    */
    cv::Scalar avg_color = cv::mean(src); // Scalar is a 4-element vector in OpenCV
    
    // Calculate the scaling factor for each channel RGB
    double scale_r = avg_color[2] / 255.0;
    double scale_g = avg_color[1] / 255.0;
    double scale_b = avg_color[0] / 255.0;

    // Apply the scaling factor to each channel for each pixel
    dst = src.clone();
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // src.at returns a Vec3b object, which is a 3-element vector that stores B, G, R values
            // and x, y are the coordinates of the pixel
            cv::Vec3b color = src.at<cv::Vec3b>(y, x);
            // uchar has 256 bits, saturate cast clamps the value to 0-255
            color[0] = cv::saturate_cast<uchar>(color[0] * scale_b);
            color[1] = cv::saturate_cast<uchar>(color[1] * scale_g);
            color[2] = cv::saturate_cast<uchar>(color[2] * scale_r);
            dst.at<cv::Vec3b>(y, x) = color; // Set the new color to the pixel
        }
    }
}


int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    cv::Mat src = cv::imread((image_dir / "casa.jpg").string(), cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cout << "Cannot open image file: " << std::endl;
        return -1;
    }

    cv::Mat dst;
    performWhiteBalance(src, dst);

    cv::imwrite((image_dir / "casa.jpg").string(), dst);

    return 0;
}