#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <filesystem>
#include <random>
#include <vector>
#include <thread>
#include "utils.cpp"

#define VIDEO_DIR "videos/"
#define file_name "sample2.mp4"
// #define file_name "sample1.wmv"

// todo Kalman para fundo
// Non max suppression
int main()
{
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path video_dir = current_file_path.parent_path() / VIDEO_DIR;

    cv::VideoCapture cap((video_dir / file_name).string());
    if (!cap.isOpened())
    {
        std::cout << "Cannot open video file: " << std::endl;
        return -1;
    }
    int frame_num = 0;
    int frame_skip = 5;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    double fps = cap.get(cv::CAP_PROP_FPS); // Get the frames per seconds of the video
    int delay = cvRound(1000.0 / fps);

    std::vector<cv::Ptr<cv::TrackerKCF>> trackers;
    std::map<cv::Ptr<cv::TrackerKCF>, int> trackerIds;
    cv::Ptr<cv::TrackerKCF> tracker;
    // std::default_random_engine generator;
    // std::uniform_int_distribution<int> distribution(1, 10000); // range of random numbers

    /*
        frame = original frame to be read from the video
        editedFrame = frame preprocessed to improve detections, not displayed, no rect drawn
        currentFrame = frame edited to be displayed (with detections and tracking)
        previousFrame = frame from the previous iteration
        diffFrame = difference between the current frame and the previous frame

    */
    cv::Mat frame, currentFrame;
    cv::Mat previousFrame, diffFrame, editedFrame, sobelFrame;
    std::vector<std::vector<cv::Point>> contours; // Find contours in the difference frame
    std::vector<cv::Rect> detections;
    std::vector<cv::Rect> motionDetections = {}; // Empty to concat with hog detections
    std::vector<double> weights;
    std::vector<double> motionWeights;
    int nextID = 1;
    double scale = 1.0;
    bool meanshift = false;

    while (true)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset the video to the beginning

        while (true)
        {
            cap >> frame; // Get new frame
            frame_num++;
            std::cout << "Starting Frame: " << frame_num << std::endl;
            if (frame.empty())
            {
                break;
            }

            /*
                Preprocess the frame to improve detections
            */
            if (frame_num % frame_skip == 0)
            {
                scale = 1.0;
                meanshift = true;
                editedFrame = frame.clone();
                // cv::Sobel(frame, sobelFrame, -1, 1, 1);
                // editedFrame = sobelFrame.clone();

                /*
                    - Upscale the frame
                    - Apply Gaussian blur
                    - Apply Sobel filter
                    - Sum filtered frames
                    - Improve contrast
                    - Find contours

                */
                improveConstrast(editedFrame);
                cv::resize(editedFrame, editedFrame, cv::Size(), scale, scale);

                // if (!previousFrame.empty())
                // {
                //     cv::resize(previousFrame, previousFrame, cv::Size(), scale, scale);
                //     cv::GaussianBlur(previousFrame, previousFrame, cv::Size(11, 11), 0);
                //     cv::absdiff(editedFrame, previousFrame, diffFrame);
                //     cv::resize(diffFrame, diffFrame, cv::Size(), 1.0 / scale, 1.0 / scale);
                //     // cv::imshow("Sobel", sobelFrame);
                //     // cv::imshow("Diff", diffFrame);
                //     // sum sobel and diff
                //     cv::GaussianBlur(diffFrame, diffFrame, cv::Size(5, 5), 0);
                //     // cv::imshow("Gaussian", diffFrame);
                //     cv::addWeighted(sobelFrame, 0.8, diffFrame, 0.8, 0, diffFrame);
                //     // sum again to improve line strength
                //     cv::addWeighted(sobelFrame, 0.8, diffFrame, 0.8, 0, diffFrame);
                //     // improve line strength
                //     // cv::threshold(diffFrame, diffFrame, 50, 255, cv::THRESH_BINARY);
                //     // cv::imshow("Sum", diffFrame);
                //     // cv::erode(diffFrame, diffFrame, cv::Mat(), cv::Point(-1, -1), 2);
                //     // cv::dilate(diffFrame, diffFrame, cv::Mat(), cv::Point(-1, -1), 2);
                //     editedFrame = diffFrame.clone();
                //     myFindContours(editedFrame, contours);
                //     myDrawContours(editedFrame, contours);
                //     // sum editedFrame and diffFrame
                //     cv::addWeighted(editedFrame, 0.6, diffFrame, 0.9, 0, editedFrame);
                //     // cv::imshow("Edited", editedFrame);
                //     cv::resize(editedFrame, editedFrame, cv::Size(), scale, scale);
                // }
                currentFrame = editedFrame.clone();

                /*
                    Start Detection
                */
            }
            if (frame_num % frame_skip == 0)
            {

                /*
                    Detection from difference frame
                */
                // hog.detectMultiScale(
                //     diffFrame,        // image
                //     motionDetections, // detections
                //     motionWeights, 0, // weights and threshold
                //     cv::Size(8, 8),   // winStride, increased for speed
                //     cv::Size(16, 16), // padding, reduced to balance speed and detection at edges
                //     1.05,             // scale, slightly adjusted for more layers without much speed loss
                //     2,                // finalThreshold, kept the same
                //     false             // useMeanshiftGrouping, set to false for speed, if accuracy is not heavily impacted
                // );

                hog.detectMultiScale(
                    currentFrame,     // image
                    detections,       // detections
                    weights, 0,       // weights and threshold
                    cv::Size(4, 4),   // winStride, increased for speed
                    cv::Size(32, 32), // padding, reduced to balance speed and detection at edges
                    1.05,             // scale, slightly adjusted for more layers without much speed loss
                    2,                // finalThreshold
                    true              // useMeanshiftGrouping, set to false for speed, if accuracy is not heavily impacted
                );

                // hog.detectMultiScale(
                //     currentFrame, // image
                //     detections,   // detections
                //     weights, 0,   // weights and threshold
                //     cv::Size(),   // winStride, increased for speed
                //     cv::Size()    // padding, reduced to balance speed and detection at edges
                // );

                // mergeDetections(detections, motionDetections); // combine the detections from the motion and the hog detector

                removeOuterRects(detections);
                mergeCloseRectangles(detections);

                trackers.clear();

                for (auto &detection : detections)
                {
                    int uniqueId = nextID++;
                    detection.x /= scale;
                    detection.y /= scale;
                    detection.width /= scale;
                    detection.height /= scale;

                    tracker = cv::TrackerKCF::create();
                    tracker->init(frame, detection);
                    trackers.push_back(tracker);

                    // trackerIds[tracker] = uniqueId; // Associate the tracker with its unique ID
                }
            }
            previousFrame = frame.clone();
            for (auto &tracker : trackers)
            {
                updateTracker(trackerIds[tracker], tracker, frame); // Update trackers in every frame
            }

            cv::imshow("Video", frame);
            if (cv::waitKey(delay) >= 0)
            {
                break;
            }
            std::cout << "Ending frame: " << frame_num << std::endl;
        }
        break;
    }

    return 0;
}