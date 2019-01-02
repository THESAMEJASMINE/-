#include "RoadSign.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
#define BLUE 0
#define GREEN 0
#define YELLOW 0
#define MINAREA 10
#define MAXAREA 1000
#define MINANGLE -1
#define MAXANGLE 1

//筛选颜色
void RoadSign::select_color(cv::Mat inputImage) {
    Mat hsv;

    Mat imageSave = inputImage.clone();

    //增加对比度和饱和度
    double alpha = 1.2; //1--3
    int beta = 5; //0--100
    //    for( int y = 0; y < inputImage.rows/3; y++ ) {
    //        for (int x = 0; x < inputImage.cols * 2 / 3; x++) {
    //            for (int c = 0; c < 3; c++) {
    //                inputImage.at<Vec3b>(y, x)[c] =
    //                        saturate_cast<uchar>(alpha * (inputImage.at<Vec3b>(y, x)[c]) + beta);
    //            }
    //        }
    //    }

    //滤波
    //    blur(inputImage, inputImage, Size(3, 3));
    //    GaussianBlur(inputImage, inputImage, Size(3, 3), 0, 0);
    //    medianBlur(inputImage, inputImage, 5);

    //RGB转为HSV图像显示
    cv::cvtColor(inputImage, hsv, cv::COLOR_BGR2HSV);

    /*****蓝色路标*****/
    //阈值分割
    //cv::inRange(hsv, cv::Scalar(214-BLUE,61-BLUE, 62-BLUE), cv::Scalar(214+BLUE, 61+BLUE,62+BLUE), blue_mask);
    cv::inRange(hsv, cv::Scalar(100+BLUE,43+BLUE, 46+BLUE), cv::Scalar(124-BLUE, 255-BLUE,255-BLUE), blue_mask);

    //创建窗口
    cv::namedWindow("blue", 0);
    cv::resizeWindow("blue", 800, 380);

    //腐蚀膨胀
    erode(blue_mask, blue_mask, getStructuringElement(MORPH_RECT, Size(3, 3)));
    dilate(blue_mask, blue_mask, getStructuringElement(MORPH_RECT, Size(5, 5)));

    //二值化
    threshold(blue_mask, blue_mask, 250, 255, CV_THRESH_BINARY);

    //显示图像
    imshow("blue", blue_mask);

    unsigned int index = 0;
    vector<vector<Point>> contours_blue;
    vector<Vec4i> hierarchy_blue;
    findContours(blue_mask, contours_blue, hierarchy_blue, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    while (index < contours_blue.size()) {
        if (contours_blue.at(index).size() >= 4 && hierarchy_blue.at(index).val[2] >= 0 && hierarchy_blue.at(index).val[3] < 0) {
            RotatedRect cnt=minAreaRect(contours_blue.at(index));
            if(cnt.angle>=MINANGLE&&cnt.angle<=MAXANGLE)
                if(cnt.size.area()>=MINAREA&&cnt.size.area()<=MAXAREA)
                    drawContours(imageSave, contours_blue, index, Scalar(255, 0, 0), 2);
        }
        index++;
    }

    /*****绿色路标*****/
    //阈值分割
    //cv::inRange(hsv, cv::Scalar(172-GREEN, 100-GREEN, 57-GREEN), cv::Scalar(172+GREEN, 100+GREEN,57+GREEN), green_mask);
    cv::inRange(hsv, cv::Scalar(35+GREEN, 43+GREEN, 46+GREEN), cv::Scalar(77-GREEN, 255-GREEN,255-GREEN), green_mask);

    //创建窗口
    cv::namedWindow("green", 0);
    cv::resizeWindow("green", 800, 380);

    //腐蚀膨胀
    erode(green_mask, green_mask, getStructuringElement(MORPH_RECT, Size(3, 3)));
    dilate(green_mask, green_mask, getStructuringElement(MORPH_RECT, Size(7, 7)));

    //二值化
    threshold(green_mask, green_mask, 250, 255, CV_THRESH_BINARY);

    //显示图像
    imshow("green", green_mask);

    index=0;
    vector<vector<Point>> contours_green;
    vector<Vec4i> hierarchy_green;
    findContours(green_mask, contours_green, hierarchy_green, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    while (index < contours_green.size()) {
        if (contours_green.at(index).size() >= 4 && hierarchy_green.at(index).val[2] >= 0 && hierarchy_green.at(index).val[3] < 0) {
            RotatedRect cnt=minAreaRect(contours_green.at(index));
            if(cnt.angle>=MINANGLE&&cnt.angle<=MAXANGLE)
                if(cnt.size.area()>=MINAREA&&cnt.size.area()<=MAXAREA)
                    drawContours(imageSave, contours_green, index, Scalar(255, 0, 0), 2);
        }
        index++;
    }

    /*****黄色路标*****/
    //阈值分割
    //cv::inRange(hsv,cv::Scalar(11-YELLOW, 43-YELLOW,46-YELLOW), cv::Scalar(34+YELLOW,255+YELLOW,255+YELLOW), yellow_mask);
    cv::inRange(hsv,cv::Scalar(11+YELLOW, 43+YELLOW,46+YELLOW), cv::Scalar(34-YELLOW,255-YELLOW,255-YELLOW), yellow_mask);

    //创建窗口
    cv::namedWindow("yellow", 0);
    cv::resizeWindow("yellow", 800, 380);

    //腐蚀膨胀
    erode(yellow_mask, yellow_mask, getStructuringElement(MORPH_RECT, Size(3,3)));
    dilate(yellow_mask, yellow_mask, getStructuringElement(MORPH_RECT, Size(5,5)));

    //二值化
    threshold(yellow_mask, yellow_mask, 245, 255, CV_THRESH_BINARY);

    //显示图像
    imshow("yellow", yellow_mask);

    index = 0;
    vector<vector<Point>> contours_yellow;
    vector<Vec4i> hierarchy_yellow;
    findContours(yellow_mask, contours_yellow, hierarchy_yellow, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    while (index < contours_yellow.size()) {
        if (contours_yellow.at(index).size() >= 4 && hierarchy_yellow.at(index).val[2] >= 0 && hierarchy_yellow.at(index).val[3] < 0) {
            RotatedRect cnt=minAreaRect(contours_yellow.at(index));
            if(cnt.angle>=MINANGLE&&cnt.angle<=MAXANGLE)
                if(cnt.size.area()>=MINAREA&&cnt.size.area()<=MAXAREA)
                    drawContours(imageSave, contours_yellow, index, Scalar(255, 0, 0), 2);
        }
        index++;
    }

    cv::namedWindow("image",0);
    imshow("image",imageSave);
}

//获取感兴趣区域
Mat RoadSign::mask(cv::Mat img) {
    cv::Mat output;
    int width, height;
    width = img.cols;
    height = img.rows;

    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

    //original的参数
    cv::Point pts[4] = {
        Point(width / 7, height * 4 / 5),//左下角开始顺时针
        Point(width / 7, height / 7),
        Point(width, height / 7),
        Point(width, height * 4 / 5)
    };

    // 创建二进制多边形，填充多边形获取感兴趣区域
    cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 255, 255));

    // 将边缘图像和掩膜进行与操作以获得输出
    cv::bitwise_and(img, mask, output);

    cv::namedWindow("ROI", 0);
    cv::resizeWindow("ROI", 800, 380);
    cv::imshow("ROI", output);

    return output;
}
