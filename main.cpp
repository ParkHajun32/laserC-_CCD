#include <opencv2/opencv.hpp>

#include <iostream>

#include <vector>

#include <cmath>

#include <string>

#include <windows.h>

#include <fstream>

#include <tuple> 

#include <iomanip> 
using namespace std;
using namespace cv;


float findDiameterOfLargestObject(Mat& image, float pixelToMmScale = 0.003) {
    // Calculate ROI as a percentage of the image size
    Size imageSize = image.size();
    Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);

    // Crop the image using the calculated ROI
    Mat croppedImage = image(roi);

    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(croppedImage, grayImage, COLOR_BGR2GRAY);

    // Invert the grayscale image since holes are darker
    Mat invertedImage;
    bitwise_not(grayImage, invertedImage);

    // Apply Gaussian Blur to reduce noise
    GaussianBlur(invertedImage, invertedImage, Size(9, 9), 2);

    // Threshold the image to get a binary image
    Mat binaryImage;
    threshold(invertedImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);

    // Find contours
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the largest object by looking for the maximum diameter
    float maxDiameter = 0;
    for (const vector<Point>& contour : contours) {
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);

        float diameter = radius * 2;
        if (diameter > maxDiameter) {
            maxDiameter = diameter; // Update the largest diameter
        }
    }

    // Convert the diameter from pixels to millimeters using the scale provided
    float diameterInMm = maxDiameter * pixelToMmScale;

    // Adjust the diameter to consider the entire image size instead of just the ROI
    diameterInMm /= (0.6 * 0.6); // Compensate for the smaller area of the ROI

    return diameterInMm; // Return the diameter in mm
}

float findRadiusOfLargestObject(Mat& image) {
    // 이미지를 그레이스케일로 변환
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // 노이즈를 줄이기 위해 Gaussian Blur 적용
    GaussianBlur(grayImage, grayImage, Size(9, 9), 2);

    // 이진화를 위한 임계값 처리
    Mat binaryImage;
    double otsu_thresh_val = threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);

    // 윤곽선 찾기
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 가장 큰 객체의 반지름을 찾음
    float maxRadius = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        if (radius > maxRadius) {
            maxRadius = radius; // 가장 큰 반지름 업데이트
        }
    }
    return maxRadius; // 가장 큰 객체의 반지름을 픽셀 단위로 반환

}


bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}
std::string createUniqueFilename(const std::string& path, const std::string& extension) {
    int counter = 0;     
    std::string filename;
    do {
        filename = path + (counter ? "(" + std::to_string(counter) + ")" : "") + extension;
        counter++;
    } while (fileExists(filename));
    return filename;
}
//float drawPinholeCircle(cv::Mat& image, const cv::Point2f& center) {
//    // 이미지 크기의 10%를 각 가장자리에서 제거합니다.
//    Size imageSize = image.size();
//    Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);
//
//    // ROI를 사용하여 이미지를 자릅니다.
//    Mat croppedImage = image(roi);
//
//    // Cropped image에서 작업을 계속합니다.
//    cv::Mat gray, blurred, thresholded;
//    cvtColor(croppedImage, gray, cv::COLOR_BGR2GRAY);
//    GaussianBlur(gray, blurred, cv::Size(9, 9), 5);
//
//    double minVal, maxVal;
//    cv::Point minLoc, maxLoc;
//    cv::minMaxLoc(blurred, &minVal, &maxVal, &minLoc, &maxLoc);
//    cv::threshold(blurred, thresholded, maxVal * 0.2, 255, cv::THRESH_BINARY_INV);
//
//    std::vector<std::vector<cv::Point>> contours;
//    cv::findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    if (!contours.empty()) {
//        float maxRadius = 0;
//        cv::Point2f maxEncCenter;
//
//        for (const auto& contour : contours) {
//            float radius;
//            cv::Point2f encCenter;
//            cv::minEnclosingCircle(contour, encCenter, radius);
//
//            if (radius > maxRadius) {
//                maxRadius = radius;
//                maxEncCenter = encCenter;
//            }
//        }
//
//        if (maxRadius > 0) {
//            // ROI 내의 상대적 위치를 원래 이미지의 절대적 위치로 변환합니다.
//            cv::Point2f absoluteCenter = cv::Point2f(maxEncCenter.x + roi.x, maxEncCenter.y + roi.y);
//            cv::circle(image, absoluteCenter, static_cast<int>(maxRadius), cv::Scalar(0, 0, 255), 2);
//
//            // Calculate diameter of the circle
//            float diameter = maxRadius * 2;
//            return diameter;
//        }
//    }
//
//    return 0; // Return 0 if no circle is found
//}

float drawPinholeCircle(cv::Mat& image, const cv::Point2f& center) {
    // Define ROI based on image size, similar to previous approach
    cv::Size imageSize = image.size();
    cv::Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);
    cv::Mat croppedImage = image(roi);

    // Convert to grayscale and apply Gaussian Blur
    cv::Mat gray, blurred, thresholded;
    cvtColor(croppedImage, gray, cv::COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, cv::Size(9, 9), 5);

    // Apply threshold, using an inverted binary to highlight features of interest
    double minVal, maxVal;
    cv::minMaxLoc(blurred, &minVal, &maxVal, nullptr, nullptr);
    threshold(blurred, thresholded, maxVal * 0.2, 255, cv::THRESH_BINARY_INV);

    // Find contours from the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        // Initialize variables to find the contour that best matches the criteria
        float maxRadius = 0;

        // Iterate through each contour to find the largest one or the one that matches certain criteria
        for (const auto& contour : contours) {
            float radius;
            cv::Point2f encCenter;
            cv::minEnclosingCircle(contour, encCenter, radius);

            // Update maxRadius if this contour's radius is larger
            if (radius > maxRadius) {
                maxRadius = radius;
            }
        }

        // Ensure the maxRadius is positive before drawing the circle
        if (maxRadius > 0) {
            // Draw the circle using the provided center and the largest radius found
            cv::circle(image, center, static_cast<int>(maxRadius), cv::Scalar(0, 0, 255), 2);

            // Calculate and return the diameter of the circle
            float diameter = maxRadius * 2;
            return diameter;
        }
    }

    return 0; // Return 0 if no suitable contour is found
}


void drawLineAndAnnotate(cv::Mat& image, cv::Point2f pt1, cv::Point2f pt2, cv::Scalar lineColor, cv::Scalar textColor) {// 라인 그리는 함수
    cv::line(image, pt1, pt2, lineColor, 2);
    float distance = cv::norm(pt1 - pt2);
    std::string text = cv::format("%.2f mm", distance);
    cv::Point2f textPos = (pt1 + pt2) * 0.5;
    cv::putText(image, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);

}
Point2f findCenter(Mat& image, bool isTarget = false) {// 중심점을 찾는 함수

    // 이미지의 크기를 얻습니다.
    Size imageSize = image.size();
    Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);

    // ROI를 사용하여 이미지를 자릅니다
    Mat croppedImage = image(roi);
    Mat gray, blurred, thresholded;
    // Cropped image를 그레이스케일로 변환
    cvtColor(croppedImage, gray, COLOR_BGR2GRAY);
    // 노이즈 감소를 위해 블러 처리
    GaussianBlur(gray, blurred, Size(9, 9), 4);


    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(blurred, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    if (!isTarget) {
        threshold(blurred, thresholded, maxVal * 0.79, 255, THRESH_BINARY);
    }
    else {
        threshold(blurred, thresholded, minVal * 1.999999, 255, THRESH_BINARY_INV);
    }
    vector<vector<Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Point2f center;

    if (!contours.empty()) {
        // 가장 큰 영역의 중심점을 찾습니다. 1860
        double maxArea = 0.0;
        vector<Point> largestContour;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }
        Moments m = moments(largestContour);
        center = Point2f(static_cast<float>(m.m10 / m.m00) + roi.x, static_cast<float>(m.m01 / m.m00) + roi.y);
    }
    return center;
}
void drawEnclosingCircle(cv::Mat& image, const cv::Point2f& center) {
    // ROI and threshold settings are inherited
    cv::Size imageSize = image.size();
    cv::Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);
    cv::Mat imageROI = image(roi);

    // Convert to grayscale and apply Gaussian Blur
    cv::Mat gray, blurred, thresholded;
    cvtColor(imageROI, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(9, 9), 5);

    // Apply threshold
    double minVal, maxVal;
    minMaxLoc(blurred, &minVal, &maxVal);
    double thresholdValue = maxVal * 0.95;
    threshold(blurred, thresholded, thresholdValue, 255, THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        // Assuming the task is to focus on the largest contour and draw a circle around the given center.
        double maxArea = 0;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
            }
        }

        // Assuming the circle to be drawn should have a radius matching the area of the largest contour
        double radius = std::sqrt(maxArea / CV_PI);

        // Draw the circle around the provided center with the calculated radius
        cv::circle(image, center, static_cast<int>(radius), cv::Scalar(0, 0, 255), 2); // Drawing with green color
    }
}

//float drawEnclosingCircle2(cv::Mat& image, const cv::Point2f& center) {
//
//    // 이미지의 크기를 얻어와서 ROI를 설정합니다.
//    cv::Size imageSize = image.size();
//    cv::Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);
//
//    // ROI에 해당하는 이미지 부분을 추출합니다.
//    cv::Mat imageROI = image(roi);
//
//    // ROI 이미지를 그레이스케일로 변환하고 블러 처리합니다.
//    cv::Mat gray, blurred, thresholded;
//    cvtColor(imageROI, gray, cv::COLOR_BGR2GRAY);
//    GaussianBlur(gray, blurred, cv::Size(9, 9), 5);
//
//    // Threshold를 적용하여 윤곽선을 찾습니다.
//    double minVal, maxVal;
//    minMaxLoc(blurred, &minVal, &maxVal);
//    double thresholdValue = maxVal * 0.65;
//    threshold(blurred, thresholded, thresholdValue, 255, cv::THRESH_BINARY);
//    std::vector<std::vector<cv::Point>> contours;
//
//
//    findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//    // ROI 내의 가장 큰 윤곽선을 찾습니다.
//    if (!contours.empty()) {
//        std::vector<cv::Point> largestContour;
//        double maxArea = 0.0;
//        for (const auto& contour : contours) {
//            double area = contourArea(contour);
//            if (area > maxArea) {
//                maxArea = area;
//                largestContour = contour;
//            }
//
//        }
//        // ROI 내에서 원을 그립니다.
//        float radius;
//        cv::Point2f encCenter;
//        cv::minEnclosingCircle(largestContour, encCenter, radius);
//        // 원의 중심을 ROI의 좌표계에서 전체 이미지의 좌표계로 변환합니다.
//        encCenter.x += roi.x;
//        encCenter.y += roi.y;
//        // 변환된 중심점을 사용하여 원을 그립니다.
//        cv::circle(image, encCenter, static_cast<int>(radius), cv::Scalar(0, 0, 255), 2);
//        float ledradius = radius;
//        return ledradius;
//    }
//}

void drawEnclosingCircle2(cv::Mat& image, const cv::Point2f& center) {
    // ROI and threshold settings are inherited
    cv::Size imageSize = image.size();
    cv::Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);
    cv::Mat imageROI = image(roi);

    // Convert to grayscale and apply Gaussian Blur
    cv::Mat gray, blurred, thresholded;
    cvtColor(imageROI, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(9, 9), 5);

    // Apply threshold
    double minVal, maxVal;
    minMaxLoc(blurred, &minVal, &maxVal);
    double thresholdValue = maxVal * 0.65;
    threshold(blurred, thresholded, thresholdValue, 255, THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        // Assuming the task is to focus on the largest contour and draw a circle around the given center.
        double maxArea = 0;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
            }
        }

        // Assuming the circle to be drawn should have a radius matching the area of the largest contour
        double radius = std::sqrt(maxArea / CV_PI);

        // Draw the circle around the provided center with the calculated radius
        cv::circle(image, center, static_cast<int>(radius), cv::Scalar(0, 0, 255), 2); // Drawing with green color
    }
}


void findAndDrawContours(Mat& image, Scalar color) {
    Mat gray, blurred, thresh;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // 이미지를 그레이스케일로 변환하고 블러 처리합니다.
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(9, 9), 2);
    // 적절한 임계값을 사용하여 이진 이미지로 변환합니다.
    threshold(blurred, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
    // 윤곽선을 찾습니다.
    findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // 윤곽선 중에서 가장 큰 것을 찾습니다.
    double maxArea = 0;
    vector<Point> maxContour;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxContour = contours[i];
        }
    }
    // 가장 큰 윤곽선을 그립니다.
    drawContours(image, vector<vector<Point>>{maxContour}, -1, color, 2);

}
void annotateCoordinateDifference(cv::Mat& image, const cv::Point2f& center1, const cv::Point2f& center2, const std::string& label, const cv::Point& position) {
    // 좌표 차이 계산
    cv::Point2f difference = center1 - center2;
    // 차이를 문자열로 변환
    std::string text = label + ": (" + std::to_string(static_cast<int>(difference.x)) + "," + std::to_string(static_cast<int>(difference.y)) + ")";

    // 이미지에 텍스트 추가
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 1;
    cv::Scalar textColor(255, 255, 255); // 흰색 텍스트
    cv::putText(image, text, position, fontFace, fontScale, textColor, thickness);
}

float convertDiameterToDistance(float diameter) {
    // Convert the diameter to distance in millimeters
    float distanceInMm = diameter * 0.003;
    return distanceInMm;
}

int main() {
    // 이미지 로드
    cv::Mat laserImage = cv::imread("C:\\Users\\USER\\Desktop\\hajun\\lasert.jpg");
    cv::Mat LEDImage = cv::imread("C:\\Users\\USER\\Desktop\\hajun\\LEDt.jpg");
    cv::Mat targetImage = cv::imread("C:\\Users\\USER\\Desktop\\hajun\\targett.jpg");
    std::string basePath = "merged_canvas";
    std::string extension = ".jpg";
    // 이미지가 제대로 로드되었는지 확인
    if (LEDImage.empty() || laserImage.empty() || targetImage.empty()) {
        std::cerr << "Error loading the images." << std::endl;
        return -1;
    }

    const float pixelToMmScale = 0.003;

    // 중심점 찾기
    cv::Point2f centerLED = findCenter(LEDImage);
    cv::Point2f centerLaser = findCenter(laserImage);
    cv::Point2f centerTarget = findCenter(targetImage, true);

    // 중심점 찍기 
    cv::circle(LEDImage, centerLED, 10, cv::Scalar(0, 0, 255), -1);
    cv::circle(laserImage, centerLaser, 10, cv::Scalar(0, 255, 0), -1);
    cv::circle(targetImage, centerTarget, 10, cv::Scalar(255, 255, 255), -1);


    // (0,0) 외곽따기
    int diameter = drawPinholeCircle(targetImage, centerTarget); // Get the diameter of the pinhole circle
    //drawPinholeCircle(targetImage, centerTarget);

    // 좌표차이 계산
    float distanceLED = cv::norm(centerLED - centerTarget);
    float distanceLaser = cv::norm(centerLaser - centerTarget);
    std::string textLEDsol = "LED to Pinhole: " + std::to_string(static_cast<int>(distanceLED)) + " pixels";
    std::string textLasersol = "Laser to Pinhole: " + std::to_string(static_cast<int>(distanceLaser)) + " pixels";




   
    drawEnclosingCircle(LEDImage, centerLED);
    drawEnclosingCircle2(laserImage, centerLaser);



    // 합성된 이미지 생성 (targetImage를 기반으로) 
    cv::Mat mergedImage = targetImage.clone();

    // LED와 Laser 중심점을 mergedImage에 추가
    cv::circle(mergedImage, centerLED, 10, cv::Scalar(0, 0, 255), -1); // LED 중심점
    cv::circle(mergedImage, centerLaser, 10, cv::Scalar(0, 255, 0), -1); // Laser 중심점


    // 거리 측정 및 텍스트 추가
    // Use a consistent scale of 0.003 mm per pixel for all distance calculations
    // Calculate distances by directly applying the scale to pixel distances
    float distanceLEDToTarget = cv::norm(centerLED - centerTarget) * pixelToMmScale;
    float distanceLaserToTarget = cv::norm(centerLaser - centerTarget) * pixelToMmScale;
    float distanceLEDToLaser = cv::norm(centerLED - centerLaser) * pixelToMmScale;



    std::string textLED = "LED light Center Distance: " + std::to_string(distanceLEDToTarget) + " mm";
    std::string textLaser = "Laser light Center Distance: " + std::to_string(distanceLaserToTarget) + " mm";
    std::string textLEDLaser = "LED <-> Laser: " + std::to_string(distanceLEDToLaser) + " mm";




    // 선 그리기
    cv::line(mergedImage, centerLED, centerLaser, cv::Scalar(255, 255, 0), 2);
    cv::line(mergedImage, centerLED, centerTarget, cv::Scalar(255, 0, 255), 2);
    cv::line(mergedImage, centerLaser, centerTarget, cv::Scalar(0, 255, 255), 2);
    cv::Mat canvas = cv::Mat::zeros(1080, 1920, mergedImage.type());
    int screenWidth = 1920;
    int screenHeight = 1080;


    // 캔버스에 들어갈 각 이미지의 크기를 계산합니다.
    int imageWidth = screenWidth / 2;
    int imageHeight = screenHeight / 2;


    // 이미지 크기 조정
    cv::resize(LEDImage, LEDImage, cv::Size(imageWidth, imageHeight));
    cv::resize(laserImage, laserImage, cv::Size(imageWidth, imageHeight));
    cv::resize(targetImage, targetImage, cv::Size(imageWidth, imageHeight));
    cv::resize(mergedImage, mergedImage, cv::Size(imageWidth, imageHeight));




    // 이미지들을 캔버스에 배치합니다.
    LEDImage.copyTo(canvas(cv::Rect(0, 0, imageWidth, imageHeight))); // 상단 왼쪽
    laserImage.copyTo(canvas(cv::Rect(imageWidth, 0, imageWidth, imageHeight))); // 상단 오른쪽
    targetImage.copyTo(canvas(cv::Rect(0, imageHeight, imageWidth, imageHeight))); // 하단 왼쪽
    mergedImage.copyTo(canvas(cv::Rect(imageWidth, imageHeight, imageWidth, imageHeight))); // 하단 오른쪽

    annotateCoordinateDifference(canvas, centerTarget, centerLED, "LED light Image Center ", cv::Point(20, 100));
    annotateCoordinateDifference(canvas, centerTarget, centerLaser, "Laser light Image Center ", cv::Point(980, 100));
    
   
    float distanceInMm = convertDiameterToDistance(diameter); // Convert diameter to distance with 0.01mm per pixel scale
    std::ostringstream distanceStream;

    distanceStream << std::fixed << std::setprecision(3) << distanceInMm; // 소수점 두 자리까지만 출력
    std::string disttarget = "diameter : " + distanceStream.str() + "mm";
    std::string pixeltext = "pixel : " + std::to_string(diameter) + " ==> pixel * 0.003 = diameter";

    // 이미지에 텍스트 추가
    cv::putText(canvas, textLED, cv::Point(980, 620), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, textLaser, cv::Point(980, 655), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, "Pinhole image Center : (0,0)", cv::Point(20, 620), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, textLEDLaser, cv::Point(980, 690), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, disttarget, cv::Point(20, 695), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, pixeltext, cv::Point(20, 655), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);

    // 이미지 표시 및 저장
    std::string uniqueFilename = createUniqueFilename(basePath, extension);
    cv::namedWindow("레이저 타격지점 분석", cv::WINDOW_NORMAL); // 창 크기 조절 가능하도록 설정
    cv::imshow("레이저 타격지점 분석", canvas); // 합성된 캔버스를 표시
    cv::waitKey(0); // 키 입력을 기다림
    cv::destroyAllWindows(); // 모든 창을 닫음
    cv::imwrite(uniqueFilename, canvas); // 캔버스 이미지를 파일로 저장
    return 0;
}