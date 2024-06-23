#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

#define IMAGE_DIR "images/q3"

// Shift quadrants to center high frequencies
void shiftDFT(cv::Mat& image) {
    int cx = image.cols / 2;
    int cy = image.rows / 2;

    cv::Mat q0(image, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(image, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(image, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // Temporary matrix for swapping
    // Swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    // Swap quadrants (Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// Manually draw pixels where the noise is estimated
void drawBlackPixels(cv::Mat& dftResult) { 
    // Coordinates and radius for the circles
    std::vector<cv::Point> centers = {{150, 177}, {185, 177}, {136, 78}, {171, 78}};
    int radius = 4;

    // Draw black filled circles on both channels of the DFT result
    for (const auto& center : centers) {
        cv::circle(dftResult, center, radius, cv::Scalar::all(0), cv::FILLED);
    }
    // Draw lines of thickness of 2 pixels from the center of the circles extending
    // 20 pixels in the vertical and horizontal directions
    cv::line(dftResult, cv::Point(150, 177), cv::Point(150, 197), cv::Scalar(0, 0, 0), 2);
    cv::line(dftResult, cv::Point(150, 177), cv::Point(170, 177), cv::Scalar(0, 0, 0), 2);

    cv::line(dftResult, cv::Point(185, 177), cv::Point(185, 197), cv::Scalar(0, 0, 0), 2);
    cv::line(dftResult, cv::Point(185, 177), cv::Point(165, 177), cv::Scalar(0, 0, 0), 2);
    cv::line(dftResult, cv::Point(185, 177), cv::Point(205, 177), cv::Scalar(0, 0, 0), 2);

    cv::line(dftResult, cv::Point(136, 78), cv::Point(136, 98), cv::Scalar(0, 0, 0), 2);
    cv::line(dftResult, cv::Point(136, 78), cv::Point(156, 78), cv::Scalar(0, 0, 0), 2);

    cv::line(dftResult, cv::Point(171, 78), cv::Point(171, 98), cv::Scalar(0, 0, 0), 2);
    cv::line(dftResult, cv::Point(171, 78), cv::Point(151, 78), cv::Scalar(0, 0, 0), 2);
    cv::line(dftResult, cv::Point(171, 78), cv::Point(191, 78), cv::Scalar(0, 0, 0), 2);

    cv::line(dftResult, cv::Point(160, 176), cv::Point(160, 254), cv::Scalar(0, 0, 0), 1);
    cv::line(dftResult, cv::Point(160, 1), cv::Point(160, 96), cv::Scalar(0, 0, 0), 1);
}

void removeSpectralNoise(cv::Mat& src, cv::Mat& dst, cv::Mat& modifiedDFT) {
    cv::Mat floatSrc;
    src.convertTo(floatSrc, CV_32F, 1.0 / 255.0); // Convert source image values to float

    cv::Mat dftResult;
    cv::dft(floatSrc, dftResult, cv::DFT_COMPLEX_OUTPUT); // Apply DFT

    /*
        Unnecessary shift, but I located the noise in a shifted image originally,
        so I kept it.
    */
    shiftDFT(dftResult); // Shift the DFT quadrants
    drawBlackPixels(dftResult); // Manual denoising
    shiftDFT(dftResult);

    modifiedDFT = dftResult.clone(); // Save dft for later reconstruction

    /*
        Calculate the magnitude
        magnitude formula is sqrt(Real(DFT(I))^2 + Imgnary(DFT(I))^2)
        this is the distance from the origin in the complex plane
    */
    cv::Mat planes[] = {cv::Mat::zeros(dftResult.size(), CV_32F), cv::Mat::zeros(dftResult.size(), CV_32F)};
    cv::split(dftResult, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // +1 bcause log(0) is undefined
    magI += cv::Scalar::all(1);
    cv::log(magI, magI); // Make log to scale down the range of magnitude values for visualization

    // Normalize to [0, 255] for visualization
    cv::normalize(magI, magI, 0, 255, cv::NORM_MINMAX);
    magI.convertTo(dst, CV_8U);
}

// Invert DFT to reconstruct image
void reconstructOriginalImage(const cv::Mat& dft, cv::Mat& dst) {
    cv::Mat inverseDft;
    cv::idft(dft, inverseDft, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Apply inverse DFT
    cv::normalize(inverseDft, dst, 0, 255, cv::NORM_MINMAX);
    dst.convertTo(dst, CV_8U);
}


int main() {
    // Load the image in grayscale
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    cv::Mat src = cv::imread((image_dir / "Questionario-3-Imagem-4.png").string(), cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "Cannot open image file." << std::endl;
        return -1;
    }

    /*
        Denoised spectrum
        Data from the DFT (after painting the noise pixels black)
        Reconstructed image from the DFT
    */
    cv::Mat denoised, modifiedDFT, reconstructed;
    removeSpectralNoise(src, denoised, modifiedDFT);
    cv::imwrite((image_dir / "denoised_spectrum.png").string(), denoised);

    reconstructOriginalImage(modifiedDFT, reconstructed);
    cv::imwrite((image_dir / "reconstructed.png").string(), reconstructed);

    return 0;
}