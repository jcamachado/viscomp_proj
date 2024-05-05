#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <filesystem>

#define IMAGE_DIR "images/"

int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    cv::Mat img = cv::imread((image_dir / "image_cam3.png").string());
    if (img.empty()) {
        std::cout << "Cannot read image file: " << std::endl;
        return -1;
    }

    // Create HOG descriptor and detector
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Detect people in the image
    std::vector<cv::Rect> detections;
    std::vector<double> weights;
    hog.detectMultiScale(img, detections, weights);

    if (detections.empty()) {
        std::cout << "No people detected in the image." << std::endl;
        return -1;
    }

    // Initialize the tracker
    cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();

    // Define bounding box for the initial object location (using the first detection)
    cv::Rect2d bbox = detections[0];

    // Initialize tracker with first frame and bounding box
    tracker->init(img, bbox);

    // Draw the detections on the image
    for (const auto& detection : detections) {
        cv::rectangle(img, detection, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Project", img);
    cv::waitKey(5000);

    return 0;
}