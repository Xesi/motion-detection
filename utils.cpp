#include "utils.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
using namespace MoveDetection;

const uint FPS = 30;


bool FileIsExist(std::string filePath) {

    bool isExist = false;
    std::ifstream fin(filePath.c_str());
 
    if(fin.is_open())
        isExist = true;
 
    fin.close();
    return isExist;
}


vector<Mat> MoveDetection::ReadFrames(const string& path, const string& prefix, const string& format) {

    vector<Mat> result;

    uint number  = 1;
    auto GetFileName = [&](uint number) {
        return path + prefix + to_string(number) + format;
    };

    string fileName = GetFileName(number++);
    while(FileIsExist(fileName)) {
        result.push_back(imread(fileName));
        fileName = GetFileName(number++);
    }

    return result;

}


uchar ComputeMedian(vector<uchar>& elements) {
    sort(elements.begin(), elements.end());
    return elements[elements.size() / 2];
}


Mat ComputeMedianColor(vector<Mat> frames, uint steps = 1) {

    // Note: Expects the image to be CV_8UC3
    Mat medianImg(frames[0].rows, frames[0].cols, CV_8UC3, Scalar(0, 0, 0));

    for(uint row = 0; row < frames[0].rows; ++row) {
        for(uint col = 0; col < frames[0].cols; ++col) {

            vector<uchar> elementsB;
            vector<uchar> elementsG;
            vector<uchar> elementsR;

            for(uint imgNumber = 0; imgNumber < frames.size(); imgNumber += steps) {

                uchar B = frames[imgNumber].at<Vec3b>(row, col)[0];
                uchar G = frames[imgNumber].at<Vec3b>(row, col)[1];
                uchar R = frames[imgNumber].at<Vec3b>(row, col)[2];

                elementsB.push_back(B);
                elementsG.push_back(G);
                elementsR.push_back(R);
            }

            medianImg.at<Vec3b>(row, col)[0] = ComputeMedian(elementsB);
            medianImg.at<Vec3b>(row, col)[1] = ComputeMedian(elementsG);
            medianImg.at<Vec3b>(row, col)[2] = ComputeMedian(elementsR);
        }
    }
    return medianImg;
}


Mat ComputeMedianGray(vector<Mat> frames, uint steps = 1) {
    Mat medianImg(frames[0].rows, frames[0].cols, CV_8UC1, Scalar(0, 0, 0));
    for(uint row = 0; row < frames[0].rows; ++row) {
        for(uint col = 0; col < frames[0].cols; ++col) {
            vector<uchar> elements;
            for(uint imgNumber = 0; imgNumber < frames.size(); imgNumber += steps) {
                elements.push_back(frames[imgNumber].at<uchar>(row, col));
            }
            medianImg.at<uchar>(row, col) = ComputeMedian(elements);
        }
    }
    return medianImg;
}


Size GetVideoSize(VideoCapture& video) {
    auto index = video.get(CAP_PROP_POS_FRAMES);
    video.set(CAP_PROP_POS_FRAMES, 0);
    Mat frame;
    video.read(frame);
    Size result = frame.size();
    video.set(CAP_PROP_POS_FRAMES, index);
    return result;
}


bool EqualColorsWithThreshold(Vec3b lhs, Vec3b rhs, uchar treshold) {
    uchar diff_B = abs(lhs[0] - rhs[0]);
    uchar diff_G = abs(lhs[1] - rhs[1]);
    uchar diff_R = abs(lhs[2] - rhs[2]);
    return diff_B < treshold && diff_G < treshold && diff_R < treshold;
}

vector<Mat> ApplyMask(const vector<Mat>& frames, const vector<Mat>& mask) {
    vector<Mat> result;
    result.reserve(frames.size());

    for(uint i = 0; i < frames.size(); ++i) {
        result.push_back(Mat::zeros(frames[i].size(), CV_8UC3));
        for(uint row = 0; row < frames[i].rows; row++) {
            for(uint col = 0; col < frames[i].cols; col++) {
                auto& lhs = frames[i].at<Vec3b>(row, col);
                auto& rhs = mask[i].at<uchar>(row, col);
                result.back().at<Vec3b>(row, col) = rhs ? lhs : Vec3b(255, 255, 255);
            }
        }
    }
    return result;
}


vector<Mat> FramesBGR2Gray(const vector<Mat>& frames) {
    vector<Mat> result(frames.size());
    for(size_t i = 0; i < frames.size(); ++i) {
        cvtColor(frames[i], result[i], COLOR_BGR2GRAY);
    }
    return result;
}


vector<Mat> FramesGaussianBlur(const vector<Mat>& frames) {
    vector<Mat> result(frames.size());
    for(size_t i = 0; i < frames.size(); ++i) {
        GaussianBlur(frames[i], result[i], {5, 5}, 0);
    }
    return result;
}



void WriteVideo(const vector<Mat> frames, const string& filename, double capPropFPS, Size size) {

    cerr << "Writing video..." << endl;
    VideoWriter writer;
    int codec = VideoWriter::fourcc('M','J','P','G');
    writer.open(filename, codec, capPropFPS, size);

    if (!writer.isOpened()) {
        throw runtime_error("Could not open the output video file for write");
    }

    for(const Mat& m : frames) {
        writer.write(m);
    }

    writer.release();
}


array<double, AvgMTSize> CalcMetrics(const vector<Mat>& motionMask, const vector<Mat>& markup) {
    array<double, AvgMTSize> result;
    result.fill(0.0);
    for(uint index = 0; index < motionMask.size(); ++index) {
        array<double, MTSize> m;
        for(uint row = 0; row < motionMask[index].rows; row++) {
            for(uint col = 0; col < motionMask[index].cols; col++) {
                uchar lhs = motionMask[index].at<uchar>(row, col);
                Vec3b rhs = markup[index].at<Vec3b>(row, col);
                if (lhs == uchar(255)) {
                    if (rhs == Vec3b(0, 255, 0) || rhs == Vec3b(0, 0, 255) || rhs == Vec3b(128, 128, 128)) {
                        m[TP] += 1;
                    } else {
                        m[FP] += 1;
                    }
                } else {
                    if (rhs == Vec3b(0, 255, 0) || rhs == Vec3b(0, 0, 255) || rhs == Vec3b(128, 128, 128)) {
                        m[FN] += 1;
                    } else if(rhs == Vec3b(0, 0, 0)) {
                        m[TN] += 1;
                    }
                }
            }
        }
        m[Acc] = (m[TP] + m[TN]) / (m[TP] + m[FP] + m[FN] + m[TN]);
        m[TPR] = m[TP] / (m[TP] + m[FN]);
        m[FPR] = m[FP] / (m[FP] + m[TN]);
        result[AvgAcc] += m[Acc];
        result[AvgTPR] += m[TPR];
        result[AvgFPR] += m[FPR];
    }
    result[AvgAcc] /= motionMask.size();
    result[AvgTPR] /= motionMask.size();
    result[AvgFPR] /= motionMask.size();

    return result;
}


Mat GetDiff(const Mat& lhs, const Mat& rhs, uchar treshold) {
    Mat result(lhs.size(), CV_8UC1);
    for(uint row = 0; row < lhs.rows; row++) {
        for(uint col = 0; col < lhs.cols; col++) {
            auto l = lhs.at<uchar>(row, col);
            auto r = rhs.at<uchar>(row, col);
            result.at<uchar>(row, col) = abs(l - r) < treshold ? 0 : 255;
        }
    }
    return result;
}


Mat CalcOpticalFlow(const Mat& lhs, const Mat& rhs) {
    Mat result;
    Mat flow(rhs.size(), CV_32FC2);
    calcOpticalFlowFarneback(lhs, rhs, flow, 0.5, 3, 15, 3, 5, 1.5, 0);
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, result, 0.0f, 255.0f, NORM_MINMAX);

    return result;
}


std::array<double, AvgMTSize> MoveDetection::ApplyMoveDetectionWithTreshold(
    const string& fileName,
    const vector<Mat>& frames,
    const vector<Mat>& markup,
    uchar treshold
) {
    cerr << "===" << endl;
    cerr << "method1 for " << fileName << " tresholdValue = " << int(treshold) << endl;

    Mat medianFrame;
    {
        Timer timer("Getting median frame");
        medianFrame = ComputeMedianColor(frames, FPS / 3);
    }

    vector<Mat> motionMask;
    motionMask.reserve(frames.size());
    {
        Timer timer("Getting motion masks");
        for(const Mat& cur : frames) {
            motionMask.push_back(Mat::zeros(cur.size(), CV_8UC1));
            for(uint row = 0; row < cur.rows; row++) {
                for(uint col = 0; col < cur.cols; col++) {
                    auto& lhs = cur.at<Vec3b>(row, col);
                    auto& rhs = medianFrame.at<Vec3b>(row, col);
                    motionMask.back().at<uchar>(row, col) = EqualColorsWithThreshold(lhs, rhs, treshold) ? 0 : 255;
                }
            }
        }
    }

    array<double, AvgMTSize> metrics = CalcMetrics(motionMask, markup);
    cerr << "AvgAcc = " << metrics[AvgAcc] << endl;
    cerr << "AvgTPR = " << metrics[AvgTPR] << endl;
    cerr << "AvgFPR = " << metrics[AvgFPR] << endl;

    vector<Mat> result;
    {
        Timer timer("Apply masks to original video");
        result = ApplyMask(frames, motionMask);
    }

    {
        Timer timer("Find and draw contours on result video");
        for(size_t i = 0; i < motionMask.size(); ++i) {
            Mat dilated;
            dilate(motionMask[i], dilated, Mat(), Point(-1,-1), 3);
            vector<Vec4i> hierarchy;
            vector<vector<Point> > contours;
            findContours(dilated, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            int idx = 0;
            if(hierarchy.empty()) {
                continue;
            }
            for(; idx >= 0; idx = hierarchy[idx][0]){
                drawContours(result[i], contours, idx, Scalar(0, 0, 255));
            }
        }
    }

    {
        Timer timer("Render result video");
        WriteVideo(result, "result/m1_t" + to_string(treshold) + "_" + fileName + "_video.avi", FPS, result[0].size());
    }

    imwrite("result/m1_t" + to_string(treshold) + "_" + fileName + "_median.jpg", medianFrame);
    imwrite("result/m1_t" + to_string(treshold) + "_" + fileName + "_frame121.jpg", result[121]);

    cerr << endl;
    return metrics;
}


std::array<double, AvgMTSize> MoveDetection::ApplyMoveDetectionWithTresholdGrayBlur(
    const string& fileName,
    const vector<Mat>& frames,
    const vector<Mat>& markup,
    uchar tresholdValue,
    bool blur
) {
    cerr << "===" << endl;
    cerr << "method2 for " << fileName << " tresholdValue = " << int(tresholdValue) << ", blur = " << blur << endl;

    vector<Mat> gray;
    {
        Timer timer("BGR2Gray for frames");
        gray = FramesBGR2Gray(frames);
    }

    if (blur) {
        Timer timer("Blur frames from video");
        vector<Mat> blured = FramesGaussianBlur(gray);
        swap(blured, gray);
    }

    Mat medianFrame;
    {
        Timer timer("Getting median frame");
        medianFrame = ComputeMedianGray(gray, FPS / 3);
    }

    vector<Mat> motionMask;
    motionMask.reserve(frames.size());
    {
        Timer timer("Getting motion masks");
        for(Mat& cur : gray) {
            motionMask.push_back(Mat::zeros(cur.size(), CV_8UC1));
            for(uint row = 0; row < cur.rows; row++) {
                for(uint col = 0; col < cur.cols; col++) {
                    auto& lhs = cur.at<uchar>(row, col);
                    auto& rhs = medianFrame.at<uchar>(row, col);
                    motionMask.back().at<uchar>(row, col) = (abs(lhs - rhs) < tresholdValue) ? 0 : 255;
                }
            }
        }
    }

    array<double, AvgMTSize> metrics = CalcMetrics(motionMask, markup);
    cerr << "AvgAcc = " << metrics[AvgAcc] << endl;
    cerr << "AvgTPR = " << metrics[AvgTPR] << endl;
    cerr << "AvgFPR = " << metrics[AvgFPR] << endl;

    vector<Mat> result;
    {
        Timer timer("Apply masks to original video");
        result = ApplyMask(frames, motionMask);
    }

    {
        Timer timer("Find and draw contours on result video");
        for(size_t i = 0; i < motionMask.size(); ++i) {
            Mat dilated;
            dilate(motionMask[i], dilated, Mat(), Point(-1,-1), 3);
            vector<Vec4i> hierarchy;
            vector<vector<Point> > contours;
            findContours(dilated, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            int idx = 0;
            if(hierarchy.empty()) {
                continue;
            }
            for(; idx >= 0; idx = hierarchy[idx][0]){
                drawContours(result[i], contours, idx, Scalar(0, 0, 255));
            }
        }
    }

    {
        Timer timer("Render result video");
        WriteVideo(result, "result/m2_t" + to_string(tresholdValue) + "_" + fileName + "_video.avi", FPS, result[0].size());
    }

    imwrite("result/m2_t" + to_string(tresholdValue) + "_" + fileName + "_median.jpg", medianFrame);
    imwrite("result/m2_t" + to_string(tresholdValue) + "_" + fileName + "_frame121.jpg", result[121]);

    return metrics;
}

std::array<double, AvgMTSize> MoveDetection::ApplyMoveDetectionWithFrameDiffrencing(
    const string& fileName,
    const vector<Mat>& frames,
    const vector<Mat>& markup,
    uchar treshold,
    uint frameCount,
    bool blur
) {
    cerr << "===" << endl;
    cerr << "method3 for " << fileName << " tresholdValue = " << int(treshold) << ", blur = " << blur << endl;

    vector<Mat> gray;
    {
        Timer timer("BGR2Gray for frames");
        gray = FramesBGR2Gray(frames);
    }

    if (blur) {
        Timer timer("Blur frames from video");
        vector<Mat> blured = FramesGaussianBlur(gray);
        swap(blured, gray);
    }

    vector<Mat> motionMask(frames.size());
    {
        Timer timer("Getting motion masks");
        for(uint i = frameCount; i < frames.size(); ++i) {
            motionMask[i] = GetDiff(gray[i], gray[i - frameCount], treshold);
        }
        for(uint i = 0; i < frameCount; ++i)
        motionMask[i] = GetDiff(gray[i + frameCount], gray[i], treshold);
    }

    array<double, AvgMTSize> metrics = CalcMetrics(motionMask, markup);
    cerr << "AvgAcc = " << metrics[AvgAcc] << endl;
    cerr << "AvgTPR = " << metrics[AvgTPR] << endl;
    cerr << "AvgFPR = " << metrics[AvgFPR] << endl;

    {
        Timer timer("Fill contoures");
        for(size_t i = 0; i < motionMask.size(); ++i) {
            Mat dilated;
            dilate(motionMask[i], dilated, Mat(), Point(-1,-1), 5);
            vector<Vec4i> hierarchy;
            vector<vector<Point> > contours;
            findContours(dilated, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            int idx = 0;
            if(hierarchy.empty()) {
                continue;
            }
            for(; idx >= 0; idx = hierarchy[idx][0]) {
                drawContours(dilated, contours, idx, 255, FILLED);
            }
            erode(dilated, motionMask[i], Mat(), Point(-1,-1), 5);
        }
    }

    vector<Mat> result;
    {
        Timer timer("Apply masks to original video");
        result = ApplyMask(frames, motionMask);
    }

    {
        Timer timer("Find and draw contours on result video");
        for(size_t i = 0; i < motionMask.size(); ++i) {
            Mat dilated;
            dilate(motionMask[i], dilated, Mat(), Point(-1,-1), 3);
            vector<Vec4i> hierarchy;
            vector<vector<Point> > contours;
            findContours(dilated, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            int idx = 0;
            if(hierarchy.empty()) {
                continue;
            }
            for(; idx >= 0; idx = hierarchy[idx][0]){
                drawContours(result[i], contours, idx, Scalar(0, 0, 255));
            }
        }
    }

    {
        Timer timer("Render result video");
        WriteVideo(result, "result/m3_t" + to_string(treshold) + "_" + fileName + "_video.avi", FPS, result[0].size());
    }
    // Display median frame
    imwrite("result/m3_t" + to_string(treshold) + "_" + fileName + "_frame121.jpg", result[121]);

    return metrics;
}


array<double, AvgMTSize> MoveDetection::ApplyMoveDetectionWithOpticalFlow(
    const string& fileName,
    const vector<Mat>& frames,
    const vector<Mat>& markup,
    uchar treshold
) {
    cerr << "===" << endl;
    cerr << "method4 for " << fileName << " tresholdValue = " << int(treshold) << endl;

    vector<Mat> gray;
    {
        Timer timer("BGR2Gray for frames");
        gray = FramesBGR2Gray(frames);
    }

    vector<Mat> flows(frames.size());
    {
        Timer timer("Find optical flows");
        for(int i = 1; i < frames.size(); ++i) {
            flows[i] = CalcOpticalFlow(gray[i - 1], gray[i]);
        }
        flows[0] = CalcOpticalFlow(gray[1], gray[0]);
    }

    vector<Mat> motionMask(frames.size());
    {
        Timer timer("Getting motion masks");
        for(uint i = 0; i < frames.size(); ++i) {
            motionMask[i] = Mat(flows[i].size(), CV_8UC1);
            for(uint row = 0; row < flows[i].rows; ++row) {
                for(uint col = 0; col < flows[i].cols; ++col) {
                    motionMask[i].at<uchar>(row, col) = flows[i].at<float>(row, col) < float(treshold) ? 0 : 255;
                }
            }
        }
    }

    array<double, AvgMTSize> metrics = CalcMetrics(motionMask, markup);
    cerr << "AvgAcc = " << metrics[AvgAcc] << endl;
    cerr << "AvgTPR = " << metrics[AvgTPR] << endl;
    cerr << "AvgFPR = " << metrics[AvgFPR] << endl;

    vector<Mat> result;
    {
        Timer timer("Apply masks to original video");
        result = ApplyMask(frames, motionMask);
    }

    {
        Timer timer("Find and draw contours on result video");
        for(size_t i = 0; i < motionMask.size(); ++i) {
            Mat dilated;
            dilate(motionMask[i], dilated, Mat(), Point(-1,-1), 3);
            vector<Vec4i> hierarchy;
            vector<vector<Point> > contours;
            findContours(dilated, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            int idx = 0;
            if(hierarchy.empty()) {
                continue;
            }
            for(; idx >= 0; idx = hierarchy[idx][0]){
                drawContours(result[i], contours, idx, Scalar(0, 0, 255));
            }
        }
    }

    {
        Timer timer("Render result video");
        WriteVideo(result, "result/m4_t" + to_string(treshold) + "_" + fileName + "_video.avi", FPS, result[0].size());
    }

    imwrite("result/m4_t" + to_string(treshold) + "_" + fileName + "_frame121.jpg", result[121]);

    return metrics;

}