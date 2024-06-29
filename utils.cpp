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
    return rect1.x >= rect2.x &&
           rect1.y >= rect2.y &&
           rect1.x + rect1.width - 3 <= rect2.x + rect2.width &&
           rect1.y + rect1.height - 3 <= rect2.y + rect2.height;
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
                detection.x + detection.width - 3 <= detection2.x + detection2.width &&
                detection.y + detection.height - 3 <= detection2.y + detection2.height)
            {
                rects.erase(std::remove(rects.begin(), rects.end(), detection2), rects.end());
            }
        }
    }
}

std::vector<cv::Rect> combineDetections(
    const std::vector<cv::Rect> &detections,
    const std::vector<cv::Rect> &motionDetections)
{
    std::vector<cv::Rect> comboDetections(detections.begin(), detections.end());
    comboDetections.insert(comboDetections.end(), motionDetections.begin(), motionDetections.end());

    // Assuming each detection is equally important, assign a constant score
    std::vector<float> scores(comboDetections.size(), 1.0f); // Example score

    float score_threshold = 0.5; // Example threshold
    float nms_threshold = 0.5;   // Example threshold
    std::vector<int> indices;

    // Apply non-maximum suppression
    cv::dnn::NMSBoxes(comboDetections, scores, score_threshold, nms_threshold, indices);

    std::vector<cv::Rect> resultDetections;
    for (int idx : indices)
    {
        resultDetections.push_back(comboDetections[idx]);
    }

    return resultDetections;
}

void drawContours(cv::Mat &frame, const std::vector<std::vector<cv::Point>> &contours)
{
    // Create a blank image with the same dimensions as the frame
    cv::Mat contoursImage = cv::Mat::zeros(frame.size(), frame.type());

    for (const auto &contour : contours)
    {
        cv::Rect boundingBox = cv::boundingRect(contour);
        // Draw the bounding box on the contoursImage instead of the frame
        cv::rectangle(contoursImage, boundingBox, cv::Scalar(0, 255, 0), 2);
    }
}

// Kalman Filter
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

// Tracker
class Tracker
{
public:
    Tracker(cv::Ptr<cv::TrackerKCF> tracker)
    {
        this->tracker = tracker;
    }

    void update(cv::Mat &frame)
    {
        cv::Rect bbox;
        // Print values for frame
        if (tracker->update(frame, bbox))
        {
            cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
            // remove overlapping rectangles
        }
    }

private:
    cv::Ptr<cv::TrackerKCF> tracker;
};
