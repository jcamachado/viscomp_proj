#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <filesystem>
#include <vector>
#include <thread>
#include "utils.cpp"

#define VIDEO_DIR "videos/"




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
    int frame_skip = 1;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = cvRound(1000.0 / fps);

    std::vector<cv::Ptr<cv::TrackerKCF>> trackers;

    while (true) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset the video to the beginning
        cv::Mat frame, currentFrame, previousFrame, diffFrame;
        // Find contours in the difference frame
        // std::vector<std::vector<cv::Point>> contours;

        while (true) {
            cap >> frame; // Get a new frame from the camera
            frame_num++;
            std::cout << "Frame number: " << frame_num << std::endl;
            if (frame.empty()) break;

            // Perform detection only on every third frame
            if (frame_num % frame_skip == 0) {
                // Upscale the frame to detect smaller objects
                double scale = 2.0; // Experiment with this value
                cv::Mat resizedFrame;
                cv::resize(frame, resizedFrame, cv::Size(), scale, scale);


                // // Set diffFrame to black
                // if (frame_num == 1) {
                //     currentFrame = frame.clone();
                //     previousFrame = currentFrame.clone();
                //     continue;
                // }
                // currentFrame = frame.clone();
                // // Subtract the current frame from the previous frame
                // subtractFrames(previousFrame, currentFrame, diffFrame);
                // previousFrame = currentFrame.clone();

                std::vector<cv::Rect> detections;
                // std::vector<cv::Rect> motionDetections;
                std::vector<double> weights;
                // std::vector<double> motionWeights;

                
                hog.detectMultiScale(
                    resizedFrame,  // image
                    detections,     // detections
                    weights, 0,     // weights
                    cv::Size(4,4),     // winStride
                    cv::Size(32,32),     // padding a better value is
                    1.05, 2, true   // scale, finalThreshold, useMeanshiftGrouping
                );

                // std::vector<cv::KalmanFilter> kalmanFilters;

                // remove fully overlapping rectangles, remove the larger one
                removeOuterRects(detections);

                for(auto& detection : detections) {
                    for(auto& detection2 : detections) {
                        if (detection == detection2) continue;
                        // if rectangule 1 is inside rectangle 2, remove rectangle 2
                        if (detection.x >= detection2.x && detection.y >= detection2.y && detection.x + detection.width <= detection2.x + detection2.width && detection.y + detection.height <= detection2.y + detection2.height) {
                            detections.erase(std::remove(detections.begin(), detections.end(), detection2), detections.end());
                        }
                    }
                }

                trackers.clear();
                for (auto& detection : detections) {
                    detection.x /= scale;
                    detection.y /= scale;
                    detection.width /= scale;
                    detection.height /= scale;
                    // cv::rectangle(frame, cv::Point(detection.x, detection.y), cv::Point(detection.x + detection.width, detection.y + detection.height), cv::Scalar(255, 0, 0), 2);

                    cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();
                    tracker->init(frame, detection);
                    trackers.push_back(tracker);
                    // Initialize Kalman filter for each detection
                    // cv::KalmanFilter kf;
                    // initializeKalmanFilter(kf, detection.x, detection.y);
                    // kalmanFilters.push_back(kf);
                    
                }

                // Predict the next state of the Kalman filter
                // for (auto& kf : kalmanFilters) {
                //     cv::Point prediction = predictKalmanFilter(kf);
                //     cv::circle(frame, prediction, 4, cv::Scalar(0, 255, 0), -1);
                // }
                for (auto& tracker : trackers) {
                    updateTracker(tracker, frame);
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