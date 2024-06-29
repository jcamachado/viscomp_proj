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
    int frame_skip = 3;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    double fps = cap.get(cv::CAP_PROP_FPS); // Get the frames per seconds of the video
    int delay = cvRound(1000.0 / fps);

    std::vector<cv::Ptr<cv::TrackerKCF>> trackers;
    std::map<cv::Ptr<cv::TrackerKCF>, int> trackerIds;
    cv::Ptr<cv::TrackerKCF> tracker;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 10000); // range of random numbers

    cv::Mat frame, currentFrame;
    // cv::Mat previousFrame, diffFrame;
    // std::vector<std::vector<cv::Point>> contours; // Find contours in the difference frame
    std::vector<cv::Rect> detections;
    std::vector<double> weights;
    int nextID = 1;

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
                Start Detection
            */
            if (frame_num % frame_skip == 0)
            {
                improveConstrast(currentFrame);
                currentFrame = frame.clone(); // Frame but editable
                /*
                    Upscale the frame to detect smaller objects
                */
                double scale = 1.0;
                // cv::resize(currentFrame, currentFrame, cv::Size(), scale, scale);

                /*
                    subtract the gaussian blurred previous frame from the
                    current frame keeping the n channels
                */
                // if (!previousFrame.empty())
                // {
                // cv::Mat blurredFrame;
                // cv::GaussianBlur(previousFrame, blurredFrame, cv::Size(11, 11), 0);
                // cv::absdiff(currentFrame, blurredFrame, diffFrame);
                // currentFrame = diffFrame.clone();

                // hog.detectMultiScale(
                //     diffFrame,        // image
                //     motionDetections, // detections
                //     motionWeights, 0, // weights and threshold
                //     cv::Size(4, 4),   // winStride, increased for speed
                //     cv::Size(32, 32), // padding, reduced to balance speed and detection at edges
                //     1.05,             // scale, slightly adjusted for more layers without much speed loss
                //     2,                // finalThreshold, kept the same
                //     false             // useMeanshiftGrouping, set to false for speed, if accuracy is not heavily impacted
                // );
                // }

                // Detection step
                if (detections.empty()) // Simplified condition for demonstration
                {
                    hog.detectMultiScale(
                        currentFrame,     // image
                        detections,       // detections
                        weights, 0,       // weights and threshold
                        cv::Size(4, 4),   // winStride, increased for speed
                        cv::Size(32, 32), // padding, reduced to balance speed and detection at edges
                        1.05,             // scale, slightly adjusted for more layers without much speed loss
                        2,                // finalThreshold
                        false             // useMeanshiftGrouping, set to false for speed, if accuracy is not heavily impacted
                    );

                    // detections = combineDetections(detections, motionDetections); // combine the detections from the motion and the hog detector

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

                        trackerIds[tracker] = uniqueId; // Associate the tracker with its unique ID
                    }
                }
                else
                {
                    // Update existing trackers
                    std::vector<int> idsToRemove;
                    for (auto &tracker : trackers)
                    {
                        cv::Rect bbox;
                        if (tracker->update(frame, bbox))
                        {
                            cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
                            putText(frame, std::to_string(trackerIds[tracker]), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 2);
                        }
                        else
                        {
                            idsToRemove.push_back(trackerIds[tracker]);
                        }
                    }

                    // Remove trackers that failed to update
                    for (int idToRemove : idsToRemove)
                    {
                        for (auto it = trackers.begin(); it != trackers.end();)
                        {
                            if (trackerIds[*it] == idToRemove)
                            {
                                trackerIds.erase(*it);   // Corrected to remove using the tracker as the key
                                it = trackers.erase(it); // Erase the tracker and update the iterator
                            }
                            else
                            {
                                ++it;
                            }
                        }
                    }
                }

                // previousFrame = currentFrame.clone();
            }
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