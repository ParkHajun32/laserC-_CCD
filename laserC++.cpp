#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <windows.h>

using namespace std;

using namespace cv;

bool drawPinholeCircle(cv::Mat& image, const cv::Point2f& center) {
    // 이미지 크기의 10%를 각 가장자리에서 제거합니다.
    int borderSizeX = image.cols * 0.3;
    int borderSizeY = image.rows * 0.3;
    cv::Rect roi(borderSizeX, borderSizeY, image.cols - 2 * borderSizeX, image.rows - 2 * borderSizeY);

    // ROI를 기준으로 이미지를 자릅니다.
    cv::Mat croppedImage = image(roi);

    // Cropped image에서 작업을 계속합니다.
    cv::Mat gray, blurred, thresholded;
    cvtColor(croppedImage, gray, cv::COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(blurred, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::threshold(blurred, thresholded, maxVal * 0.2, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        float maxRadius = 0;
        cv::Point2f maxEncCenter;
        for (const auto& contour : contours) {
            float radius;
            cv::Point2f encCenter;
            cv::minEnclosingCircle(contour, encCenter, radius);

            if (radius > maxRadius) {
                maxRadius = radius;
                maxEncCenter = encCenter;
            }
        }

        if (maxRadius > 0) {
            // ROI 내의 상대적 위치를 원래 이미지의 절대적 위치로 변환합니다.
            cv::Point2f absoluteCenter = cv::Point2f(maxEncCenter.x + roi.x, maxEncCenter.y + roi.y);
            cv::circle(image, absoluteCenter, static_cast<int>(maxRadius), cv::Scalar(0, 0, 255), 2);
            return true;
        }
    }
    return false;
}


// 라인 그리는 함수
void drawLineAndAnnotate(cv::Mat& image, cv::Point2f pt1, cv::Point2f pt2, cv::Scalar lineColor, cv::Scalar textColor) {
    cv::line(image, pt1, pt2, lineColor, 2);
    float distance = cv::norm(pt1 - pt2);
    std::string text = cv::format("%.2f mm", distance);
    cv::Point2f textPos = (pt1 + pt2) * 0.5;
    cv::putText(image, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
}


// 중심점을 찾는 함수
Point2f findCenter(Mat& image, bool isTarget = false) {
    // 이미지의 크기를 얻습니다.
    Size imageSize = image.size();
    Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);

    // ROI를 사용하여 이미지를 자릅니다.
    Mat croppedImage = image(roi);

    // 이제 croppedImage에 대해 중심점 찾기 로직을 적용합니다.
    // 기존 코드의 나머지 부분은 croppedImage에 대해 실행됩니다.

    Mat gray, blurred, thresholded;

    // Cropped image를 그레이스케일로 변환
    cvtColor(croppedImage, gray, COLOR_BGR2GRAY);

    // 노이즈 감소를 위해 블러 처리
    GaussianBlur(gray, blurred, Size(9, 9), 2);

    // 나머지 코드...

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(blurred, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    if (!isTarget) {
        threshold(blurred, thresholded, maxVal * 0.8, 255, THRESH_BINARY);
    }
    else {
        threshold(blurred, thresholded, minVal * 1.9, 255, THRESH_BINARY_INV);
    }

    vector<vector<Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point2f center;
    if (!contours.empty()) {
        // 가장 큰 영역의 중심점을 찾습니다.
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
    // 이미지의 크기를 얻어와서 ROI를 설정합니다.
    cv::Size imageSize = image.size();
    cv::Rect roi(imageSize.width * 0.2, imageSize.height * 0.2, imageSize.width * 0.6, imageSize.height * 0.6);

    // ROI에 해당하는 이미지 부분을 추출합니다.
    cv::Mat imageROI = image(roi);

    // ROI 이미지를 그레이스케일로 변환하고 블러 처리합니다.
    cv::Mat gray, blurred, thresholded;
    cvtColor(imageROI, gray, cv::COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

    // Threshold를 적용하여 윤곽선을 찾습니다.
    double minVal, maxVal;
    minMaxLoc(blurred, &minVal, &maxVal);

    double thresholdValue = maxVal * 0.8;
    threshold(blurred, thresholded, thresholdValue, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // ROI 내의 가장 큰 윤곽선을 찾습니다.
    if (!contours.empty()) {
        std::vector<cv::Point> largestContour;
        double maxArea = 0.0;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        // ROI 내에서 원을 그립니다.
        float radius;
        cv::Point2f encCenter;
        cv::minEnclosingCircle(largestContour, encCenter, radius);

        // 원의 중심을 ROI의 좌표계에서 전체 이미지의 좌표계로 변환합니다.
        encCenter.x += roi.x;
        encCenter.y += roi.y;

        // 변환된 중심점을 사용하여 원을 그립니다.
        cv::circle(image, encCenter, static_cast<int>(radius), cv::Scalar(0, 0, 255), 2);
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
    std::string text = label + ": (" + std::to_string(static_cast<int>(difference.x))+"," + std::to_string(static_cast<int>(difference.y)) + ")";

    // 이미지에 텍스트 추가
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 1;
    cv::Scalar textColor(255, 255, 255); // 흰색 텍스트

    cv::putText(image, text, position, fontFace, fontScale, textColor, thickness);
}

int main() {
    // 이미지 로드
    cv::Mat laserImage = cv::imread("C:\\Users\\USER\\Desktop\\hajun\\LEDUI.jpg");
    cv::Mat LEDImage = cv::imread("C:\\Users\\USER\\Desktop\\hajun\\laserUI.jpg");
    cv::Mat targetImage = cv::imread("C:\\Users\\USER\\Desktop\\hajun\\targetUI.jpg");


    // 이미지가 제대로 로드되었는지 확인
    if (LEDImage.empty() || laserImage.empty() || targetImage.empty()) {
        std::cerr << "Error loading the images." << std::endl;
        return -1;
    }

    // 중심점 찾기
    cv::Point2f centerLED = findCenter(LEDImage);
    cv::Point2f centerLaser = findCenter(laserImage);
    cv::Point2f centerTarget = findCenter(targetImage, true);
    cv::circle(LEDImage, centerLED, 10, cv::Scalar(0, 255, 0), -1);
    cv::circle(laserImage, centerLaser, 10, cv::Scalar(0, 0, 255), -1);
    cv::circle(targetImage, centerTarget, 10, cv::Scalar(255, 255, 255), -1);
    drawPinholeCircle(targetImage, centerTarget);


    float distanceLED = cv::norm(centerLED - centerTarget);
    float distanceLaser = cv::norm(centerLaser - centerTarget);
    std::string textLEDsol = "LED to Pinhole: " + std::to_string(static_cast<int>(distanceLED)) + " pixels";
    std::string textLasersol = "Laser to Pinhole: " + std::to_string(static_cast<int>(distanceLaser)) + " pixels";

    // 합성된 이미지 생성 (targetImage를 기반으로)
    cv::Mat mergedImage = targetImage.clone();
    drawEnclosingCircle(LEDImage, centerLED);
    drawEnclosingCircle(laserImage, centerLaser);
   

    // LED와 Laser 중심점을 mergedImage에 추가
    cv::circle(mergedImage, centerLED, 10, cv::Scalar(0, 255, 0), -1); // LED 중심점
    cv::circle(mergedImage, centerLaser, 10, cv::Scalar(0, 0, 255), -1); // Laser 중심점


    // 거리 측정 및 텍스트 추가
    float pixelToMMRatio = 4 / cv::norm(centerTarget); // 화면의 크기에 따라 조정이 필요할 수 있음
    float distanceLEDToTarget = cv::norm(centerLED - centerTarget) * pixelToMMRatio;
    float distanceLaserToTarget = cv::norm(centerLaser - centerTarget) * pixelToMMRatio;
    float distanceLEDToLaser = cv::norm(centerLED - centerLaser) * pixelToMMRatio;

    std::string textLED = "LED light Center Distance: " + std::to_string(distanceLEDToTarget) + " mm";
    std::string textLaser = "Laser light Center Distance: " + std::to_string(distanceLaserToTarget) + " mm";
    std::string textLEDLaser = "LED <-> Laser: " + std::to_string(distanceLEDToLaser) + " mm";

   

    // 선 그리기
    //cv::line(mergedImage, centerLED, centerLaser, cv::Scalar(255, 255, 0), 2);
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

    // 이미지에 텍스트 추가
    cv::putText(canvas, textLED, cv::Point(980, 620), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, textLaser, cv::Point(980, 655), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, "Pinhole image Center : (0,0)", cv::Point(20, 620), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, textLEDLaser, cv::Point(980, 690), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);

 


    // 이미지 표시 및 저장
    cv::namedWindow("레이저 타격지점 분석", cv::WINDOW_NORMAL); // 창 크기 조절 가능하도록 설정
    cv::imshow("레이저 타격지점 분석", canvas); // 합성된 캔버스를 표시
    cv::waitKey(0); // 키 입력을 기다림
    cv::destroyAllWindows(); // 모든 창을 닫음

    cv::imwrite("merged_canvas.jpg", canvas); // 캔버스 이미지를 파일로 저장

    return 0;


}


