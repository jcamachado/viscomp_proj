#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <iostream>

#define IMAGE_DIR "images/q4"

void applySIFT(cv::Mat &img1, cv::Mat &img2) {
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * 0.15f; // Good matches are the top 5% matches 
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Extract location of good matches
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    /*
        Quick reminder of RAMSAC:
        - Randomly select 4 points from the set of matches 
        - Check how they approximate with each other between the two images (matches)
        - If the approximation is good, then the 4 points are inliers
    */
    // Find homography using RANSAC
    cv::Mat mask;
    cv::findHomography(points1, points2, cv::RANSAC, 5.0, mask); // 5.0 is the RANSAC reprojection threshold

    // Use mask to select the inlier matches
    std::vector<cv::DMatch> inlierMatches;
    for (size_t i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i)) {
            inlierMatches.push_back(matches[i]);
        }
    }

    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, inlierMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches", imgMatches);
    cv::waitKey(0);
}

void createVGG16() {

}


int main() {
    // Load the image in grayscale
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path image_dir = current_file_path.parent_path() / IMAGE_DIR;

    // cv::Mat img1 = cv::imread((image_dir / "Questionario-4-Bricks1.jpg").string());
    // cv::Mat img2 = cv::imread((image_dir / "Questionario-4-Bricks2.jpg").string());

    cv::Mat img1 = cv::imread((image_dir / "Questionario-4-Building1.jpg").string());
    cv::Mat img2 = cv::imread((image_dir / "Questionario-4-Building2.jpg").string());


    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not load image" << std::endl;
        return 1;
    }

    // applySIFT(img1, img2);
    createVGG16();
    
    return 0;
}