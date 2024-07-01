#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <filesystem>
#include <vector>
#include <thread>

void findContours(cv::Mat &frame, std::vector<std::vector<cv::Point>> &contours)
{
    cv::Mat frameCopy = frame.clone();
    cv::findContours(frameCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

int getRectangleCenterX(const cv::Rect &rect)
{
    return rect.x + rect.width / 2;
}

int getRectangleCenterY(const cv::Rect &rect)
{
    return rect.y + rect.height / 2;
}

// if rectangles are almost overlapping by 3 pixels, get the average of the 2 rectangles
cv::Point getAverageCenter(const cv::Rect &rect1, const cv::Rect &rect2)
{
    int x1 = getRectangleCenterX(rect1);
    int y1 = getRectangleCenterY(rect1);
    int x2 = getRectangleCenterX(rect2);
    int y2 = getRectangleCenterY(rect2);

    return cv::Point((x1 + x2) / 2, (y1 + y2) / 2);
}

// check if 2 rectangles are almost overlapping by 3 pixels
bool isOverlapping(const cv::Rect &rect1, const cv::Rect &rect2)
{
    int margin = 10;
    return rect1.x >= rect2.x &&
           rect1.y >= rect2.y &&
           rect1.x + rect1.width - margin <= rect2.x + rect2.width &&
           rect1.y + rect1.height - margin <= rect2.y + rect2.height;
}

// if rectangles are almost overlapping by 3 pixels, merge the 2 rectangles
void mergeCloseRectangles(std::vector<cv::Rect> detections)
{
    for (auto &detection : detections)
    {
        for (auto &detection2 : detections)
        {
            if (detection == detection2)
                continue;
            if (isOverlapping(detection, detection2))
            {
                detection.x = std::min(detection.x, detection2.x);
                detection.y = std::min(detection.y, detection2.y);
                detection.width = std::max(detection.x + detection.width, detection2.x + detection2.width) - detection.x;
                detection.height = std::max(detection.y + detection.height, detection2.y + detection2.height) - detection.y;
                detections.erase(std::remove(detections.begin(), detections.end(), detection2), detections.end());
            }
        }
    }
}

void improveConstrast(cv::Mat &frame)
{
    double alpha = 2.0; // Contrast control (1.0-3.0)
    int beta = 0;       // Brightness control (0-100)
    cv::Mat contrastFrame;
    frame.convertTo(contrastFrame, -1, alpha, beta);
}

void initializeKalmanFilter(cv::KalmanFilter &kf, double x, double y)
{
    kf.init(4, 2, 0); // State size, measurement size, control size

    // State transition matrix (A)
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                           0, 1, 0, 1,
                           0, 0, 1, 0,
                           0, 0, 0, 1);
    // Measurement matrix (H)
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                            0, 1, 0, 0);

    // Process noise covariance (Q)
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
    // Measurement noise covariance (R)
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
    // Error covariance (P)
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    // Initial state
    kf.statePost.at<float>(0) = x;
    kf.statePost.at<float>(1) = y;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;
}

cv::Point predictKalmanFilter(cv::KalmanFilter &kf)
{
    cv::Mat prediction = kf.predict();
    return cv::Point(prediction.at<float>(0), prediction.at<float>(1));
}

void updateKalmanFilter(cv::KalmanFilter &kf, cv::Point meas)
{
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
    measurement.at<float>(0) = meas.x;
    measurement.at<float>(1) = meas.y;

    kf.correct(measurement);
}

void updateTracker(int id, cv::Ptr<cv::TrackerKCF> tracker, cv::Mat &frame)
{
    cv::Rect bbox;
    if (tracker->update(frame, bbox))
    {
        cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
        putText(frame, std::to_string(id), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 2);
    }
}

void removeOuterRects(std::vector<cv::Rect> &rects)
{
    for (auto &detection : rects)
    {
        for (auto &detection2 : rects)
        {
            if (detection == detection2)
                continue;
            // if rectangule 1 is inside rectangle 2, remove rectangle 2
            if (detection.x >= detection2.x &&
                detection.y >= detection2.y &&
                detection.x + detection.width - 10 <= detection2.x + detection2.width &&
                detection.y + detection.height - 10 <= detection2.y + detection2.height)
            {
                rects.erase(std::remove(rects.begin(), rects.end(), detection2), rects.end());
            }
        }
    }
}

void mergeDetections(std::vector<cv::Rect> &detections, const std::vector<cv::Rect> &motionDetections)
{
    for (const auto &detection : motionDetections)
    {
        detections.push_back(detection);
    }
}

// Identify the contours in the frame
void myFindContours(cv::Mat &frame, std::vector<std::vector<cv::Point>> &contours)
{
    cv::Mat gray, edged;
    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // Apply Canny edge detection or another binarization method
    cv::Canny(gray, edged, 100, 200); // These thresholds can be adjusted based on your specific needs

    // Use the binary image for finding contours
    findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

// Draw the contours on the frame from points
void myDrawContours(cv::Mat &frame, const std::vector<std::vector<cv::Point>> &contours)
{
    // Create a temporary vector of vectors to hold each contour
    std::vector<std::vector<cv::Point>> tempContours;

    // Since the original function seems to be designed to handle individual points as contours,
    // we need to adapt it to the expected input of cv::drawContours by wrapping each point in a vector.
    for (const auto &contour : contours)
    {
        // Wrap the individual contour point in a vector
        std::vector<cv::Point> tempContour = {contour};

        // Add the wrapped contour to the collection of contours
        tempContours.push_back(tempContour);
    }

    // Now, draw each contour on the frame
    // Note: The third parameter is the contour index. To draw all contours, it is set to -1.
    cv::drawContours(frame, tempContours, -1, cv::Scalar(0, 255, 0), 2);

    // imshow
    // cv::imshow("Contours", frame);
}

// void myDrawContours(cv::Mat &frame, const std::vector<cv::Rect> &detections)
// {
//     for (const auto &detection : detections)
//     {
//         cv::rectangle(frame, detection, cv::Scalar(0, 255, 0), 2);
//     }
// }

// Kalman Filter // Reference unused
class KalmanFilter
{
public:
    KalmanFilter(double x, double y)
    {
        kf.init(4, 2, 0); // State size, measurement size, control size

        // State transition matrix (A)
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                               0, 1, 0, 1,
                               0, 0, 1, 0,
                               0, 0, 0, 1);
        // Measurement matrix (H)
        kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                                0, 1, 0, 0);

        // Process noise covariance (Q)
        kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
        // Measurement noise covariance (R)
        kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
        // Error covariance (P)
        kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

        // Initial state
        kf.statePost.at<float>(0) = x;
        kf.statePost.at<float>(1) = y;
        kf.statePost.at<float>(2) = 0;
        kf.statePost.at<float>(3) = 0;
    }

    cv::Point predict()
    {
        cv::Mat prediction = kf.predict();
        return cv::Point(prediction.at<float>(0), prediction.at<float>(1));
    }

    void update(cv::Point meas)
    {
        cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
        measurement.at<float>(0) = meas.x;
        measurement.at<float>(1) = meas.y;

        kf.correct(measurement);
    }

private:
    cv::KalmanFilter kf;
};
/*
    Reference for id tracking
    for (auto &detection : detections)
        {
            bool matched = false;
            for (auto &[id, tracker] : trackers)
            {
                cv::Rect2d bbox;
                if (tracker->update(frame, bbox))
                {
                    if ((cv::Rect(bbox) & detection).area() > 0) // Simple intersection check
                    {
                        matched = true;
                        break;
                    }
                }
            }

            if (!matched)
            {
                auto tracker = cv::TrackerKCF::create();
                tracker->init(frame, cv::Rect2d(detection));
                trackers[nextID++] = tracker;
            }
        }

        // Draw tracked objects
        for (auto &[id, tracker] : trackers)
        {
            cv::Rect2d bbox;
            if (tracker->update(frame, bbox))
            {
                cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, std::to_string(id), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
            }
        }

*/