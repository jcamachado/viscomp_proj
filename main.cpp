#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <filesystem>
#include <vector>
#include <thread>

#define VIDEO_DIR "videos/"

void updateTracker(cv::Ptr<cv::TrackerKCF> tracker, cv::Mat& frame) {
    cv::Rect bbox;
    if (tracker->update(frame, bbox)) {
        cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
    }
}


// todo Kalman para fundo
// Non max suppression
// Mean shifting
int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path video_dir = current_file_path.parent_path() / VIDEO_DIR;

    cv::VideoCapture cap((video_dir / "sample2.mp4").string());
    if (!cap.isOpened()) {
        std::cout << "Cannot open video file: " << std::endl;
        return -1;
    }
    int frame_num = 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = cvRound(1000.0 / fps);

    std::vector<cv::Ptr<cv::TrackerKCF>> trackers;

    while (true) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        while (true) {
            frame_num++;
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            std::cout << "Frame number: " << frame_num << std::endl;

            // Perform detection only on every third frame
            if (frame_num % 3 == 1) {
                std::vector<cv::Rect> detections;
                std::vector<double> weights;
                hog.detectMultiScale(frame, detections, weights);

                // Clear existing trackers and initialize new ones for the detected objects
                trackers.clear();
                for (const auto& detection : detections) {
                    cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();
                    tracker->init(frame, detection);
                    trackers.push_back(tracker);
                }
            }

            // Update trackers in every frame
            for (auto& tracker : trackers) {
                updateTracker(tracker, frame);
            }

            cv::imshow("Video", frame);
            if (cv::waitKey(delay) >= 0) {
                break; // Exit the inner loop to finish the program
            }
        }
        break; // Exit the outer loop to finish the program
    }

    return 0;
}