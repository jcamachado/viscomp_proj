#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <filesystem>

#define IMAGE_DIR "images/"

// void performWhiteBalance(const cv::Mat& src, cv::Mat& dst) {
//     // cv::xphoto::createSimpleWB()->balanceWhite(src, dst); // Default from OpenCV
//     /*
//         Calculate the average color of the whole image
//         mean reaturns a Scalar object with 4 elements: B, G, R, A
//     */
//     // ROI (Region of Interest), Chosen manually
//     cv::Rect roi(100, 100, 30, 30); //Define a 30x30 square at the coordinate (100, 100)

//     // If ROI inbounds
//     if ((roi.x + roi.width <= src.cols) && (roi.y + roi.height <= src.rows)) {
//         // Calculate the average color of the ROI
//         cv::Scalar avg_color = cv::mean(src(roi));

//         // Calculate the scaling factor for each channel RGB
//         double scale_r = avg_color[2] / 255.0;
//         double scale_g = avg_color[1] / 255.0;
//         double scale_b = avg_color[0] / 255.0;

//         // Apply the scaling factor to each channel for each pixel
//         dst = src.clone();
//         for (int y = 0; y < src.rows; y++) {
//             for (int x = 0; x < src.cols; x++) {
//                 cv::Vec3b color = src.at<cv::Vec3b>(y, x);
//                 color[0] = cv::saturate_cast<uchar>(color[0] * scale_b);
//                 color[1] = cv::saturate_cast<uchar>(color[1] * scale_g);
//                 color[2] = cv::saturate_cast<uchar>(color[2] * scale_r);
//                 dst.at<cv::Vec3b>(y, x) = color; // Set the new color to the pixel
//             }
//         }
//     } else {
//         std::cout << "ROI is out of image bounds." << std::endl;
//     }
// }
// convertion from http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
// values reference from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
// values for M in sRGB D65 to XYZ ([M][r g b]t = [X Y Z]t)

cv::Mat rgb2xyz(const cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC3);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b color = src.at<cv::Vec3b>(y, x);
            // Convert the color from 0-255 to 0-1
            double r = color[2] / 255.0; 
            double g = color[1] / 255.0;
            double b = color[0] / 255.0;
            
            r = (r <= 0.04045) ? r / 12.92 : pow((r + 0.055) / 1.055, 2.4);
            g = (g <= 0.04045) ? g / 12.92 : pow((g + 0.055) / 1.055, 2.4);
            b = (b <= 0.04045) ? b / 12.92 : pow((b + 0.055) / 1.055, 2.4);

            // Convert the color from sRGB to XYZ (D65)
            double _x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
            double _y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
            double _z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

            dst.at<cv::Vec3f>(y, x) = cv::Vec3f(_x, _y, _z);
        }
    }
    return dst;
}

/*
    XYZ to L*a*b* conversion
    http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
    Tristimumulus values for D65 https://en.wikipedia.org/wiki/Standard_illuminant
    by default, using D65 white:  white_x = 0.95047, white_y = 1.0, white_z = 1.08883
*/
cv::Mat xyz2lab(const cv::Mat& src, 
            float white_x = 0.95047, 
            float white_y = 1.0, 
            float white_z = 1.08883) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC3);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3f color = src.at<cv::Vec3f>(y, x);
            // normalize the XYZ values from white values
            double nx = color[0] / white_x;
            double ny = color[1] / white_y;
            double nz = color[2] / white_z;

            // Convert the color from XYZ to L*a*b*
            // f is a non-linear transformation from XYZ to Lab (by CIE)
            // Constants: epsilon = 216/24389 = 0.008856;  kappa = 24389/27 = 903.3
            double fx = nx > 0.008856 ? pow(nx, 1.0 / 3.0) : 7.787 * nx + 16.0 / 116.0;
            double fy = ny > 0.008856 ? pow(ny, 1.0 / 3.0) : 7.787 * ny + 16.0 / 116.0;
            double fz = nz > 0.008856 ? pow(nz, 1.0 / 3.0) : 7.787 * nz + 16.0 / 116.0;

            double l = 116.0 * fy - 16.0;
            double a = 500.0 * (fx - fy);
            double b = 200.0 * (fy - fz);

            dst.at<cv::Vec3f>(y, x) = cv::Vec3f(l, a, b);
        }
    }
    return dst;
}


// lab to XYZ
cv::Mat lab2xyz(const cv::Mat& src, 
            float white_x = 0.95047, 
            float white_y = 1.0, 
            float white_z = 1.08883) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC3);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3f color = src.at<cv::Vec3f>(y, x);
            // Convert the color from L*a*b* to XYZ
            double fy = (color[0] + 16.0) / 116.0;
            double fx = color[1] / 500.0 + fy;
            double fz = fy - color[2] / 200.0;

            // double _x = white_x * (fx > 0.206893034 ? pow(fx, 3.0) : (fx - 16.0 / 116.0) / 7.787);
            // double _y = white_y * (fy > 0.206893034 ? pow(fy, 3.0) : (fy - 16.0 / 116.0) / 7.787);
            // double _z = white_z * (fz > 0.206893034 ? pow(fz, 3.0) : (fz - 16.0 / 116.0) / 7.787);

            double _x = (fx > 0.206893034) ? pow(fx, 3.0) : (fx * 116.0 - 16.0) / 903.3;
            double _y = (fy > 0.206893034) ? pow(fy, 3.0) : (fy * 116.0 - 16.0) / 903.3;
            double _z = (fz > 0.206893034) ? pow(fz, 3.0) : (fz * 116.0 - 16.0) / 903.3;

            dst.at<cv::Vec3f>(y, x) = cv::Vec3f(_x, _y, _z);
        }
    }
    return dst;
}

// XYZ to RGB
cv::Mat xyz2rgb(const cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3f color = src.at<cv::Vec3f>(y, x);
            // Convert the color from XYZ to sRGB (D65) crude approximation
            double r = color[0] * 3.2404542 + color[1] * -1.5371385 + color[2] * -0.4985314;
            double g = color[0] * -0.9692660 + color[1] * 1.8760108 + color[2] * 0.0415560;
            double b = color[0] * 0.0556434 + color[1] * -0.2040259 + color[2] * 1.0572252;

            // Convert the color from linear to gamma corrected sRGB
            r = (r <= 0.0031308) ? 12.92 * r : 1.055 * pow(r, 1.0 / 2.4) - 0.055;
            g = (g <= 0.0031308) ? 12.92 * g : 1.055 * pow(g, 1.0 / 2.4) - 0.055;
            b = (b <= 0.0031308) ? 12.92 * b : 1.055 * pow(b, 1.0 / 2.4) - 0.055;

            // Convert the color from 0-1 to 0-255
            r = cv::saturate_cast<uchar>(r * 255.0);
            g = cv::saturate_cast<uchar>(g * 255.0);
            b = cv::saturate_cast<uchar>(b * 255.0);

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    return dst;
}

// RGB to D50 to RGB
cv::Mat rgb2labD50(const cv::Mat& src) {
    cv::Mat xyz = rgb2xyz(src);
    cv::Mat lab = xyz2lab(xyz, 0.96422, 1.0, 0.82521);
    cv::Mat xyz2 = lab2xyz(lab, 0.96422, 1.0, 0.82521);
    cv::Mat rgb = xyz2rgb(xyz2);
    return rgb;
}

// RGB to D55 to RGB
cv::Mat rgb2labD55(const cv::Mat& src) {
    cv::Mat xyz = rgb2xyz(src);
    cv::Mat lab = xyz2lab(xyz, 0.95682, 1.0, 0.92149);
    cv::Mat xyz2 = lab2xyz(lab, 0.95682, 1.0, 0.92149);
    cv::Mat rgb = xyz2rgb(xyz2);
    return rgb;
}

// RGB to D65 to RGB
cv::Mat rgb2lab(const cv::Mat& src) {
    cv::Mat xyz = rgb2xyz(src);
    cv::Mat lab = xyz2lab(xyz);
    cv::Mat xyz2 = lab2xyz(lab);
    cv::Mat rgb = xyz2rgb(xyz2);
    return rgb;
}

// RGB to D75 to RGB
cv::Mat rgb2labD75(const cv::Mat& src) {
    cv::Mat xyz = rgb2xyz(src);
    cv::Mat lab = xyz2lab(xyz, 0.94972, 1.0, 1.22638);
    cv::Mat xyz2 = lab2xyz(lab, 0.94972, 1.0, 1.22638);
    cv::Mat rgb = xyz2rgb(xyz2);
    return rgb;
}


int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    cv::Mat src = cv::imread((image_dir / "cartas.jpg").string(), cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cout << "Cannot open image file: " << std::endl;
        return -1;
    }

    // cv::Mat dst;
    // performWhiteBalance(src, dst);
    // cv::imwrite((image_dir / "cartas_wb0.png").string(), dst);

    // D50
    cv::Mat dstD50 = rgb2labD50(src);
    cv::imwrite((image_dir / "cartas_wb50.png").string(), dstD50);

    // D55
    cv::Mat dstD55 = rgb2labD55(src);
    cv::imwrite((image_dir / "cartas_wb55.png").string(), dstD55);

    // D65
    cv::Mat dstD65 = rgb2lab(src);
    cv::imwrite((image_dir / "cartas_wb65.png").string(), dstD65);

    // D75
    cv::Mat dstD75 = rgb2labD75(src);
    cv::imwrite((image_dir / "cartas_wb75.png").string(), dstD75);
    
    return 0;
}