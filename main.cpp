#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <filesystem>
#include <vector>

#define VIDEO_DIR "videos/"

int main() {
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path video_dir = current_file_path.parent_path() / VIDEO_DIR;

    cv::VideoCapture cap((video_dir / "sample2.mp4").string());
    if (!cap.isOpened()) {
        std::cout << "Cannot open video file: " << std::endl;
        return -1;
    }
    int frame_num = 0;

    // Create HOG descriptor and detector
    cv::HOGDescriptor hog;
    // TODO Customize the people detector for improved accuracy
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Frame rate of the video
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = cvRound(100.0 / fps);

    while (true) {  // Outer loop for playing the video in a loop
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // Reset the video to the first frame

        //TODO improve tracking accuracy and performance
        //TODO add id to each tracked person
        while (true) {  // Inner loop for playing the video
            frame_num++;
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            std::cout << "Frame number: " << frame_num << std::endl;

            // Detect people in the current frame
            std::vector<cv::Rect> detections;
            std::vector<double> weights;
            hog.detectMultiScale(frame, detections, weights);

            if (detections.empty()) {
                std::cout << "No people detected in the image." << std::endl;
            }

            // Initialize a tracker for each person
            std::vector<cv::Ptr<cv::TrackerKCF>> trackers;
            for (const auto& detection : detections) {
                cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();
                tracker->init(frame, detection);
                trackers.push_back(tracker);
            }

            // Update each tracker and draw the bounding box
            for (size_t i = 0; i < trackers.size(); ++i) {
                cv::Rect bbox;
                if (trackers[i]->update(frame, bbox)) {
                    cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
                }
            }

            cv::imshow("Video", frame);
            if (cv::waitKey(delay) >= 0) {  // Delay for maintaining the original frame rate
                return 0;  // Exit the program if the user presses a key
            }
        }
    }

    return 0;
}