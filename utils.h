#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>
#include <array>


namespace MoveDetection {

    enum MetricType {
        TP = 0,
        TN = 1,
        FP = 2,
        FN = 3,
        Acc = 4,
        TPR = 5,
        FPR = 6,

        MTSize = 7
    };

    enum AvgMetricType {
        AvgAcc = 0,
        AvgTPR = 1,
        AvgFPR = 2,

        AvgMTSize = 3
    };

    class Timer {
    public:
        explicit Timer(const std::string& message)
            : message(message) {
            start = clock();
        }
        ~Timer() {
            clock_t end = clock();
            double seconds = (double)(end - start) / CLOCKS_PER_SEC;
            std::cerr << message << " run for " << seconds << " seconds." <<std::endl;
        }
    private:
        std::string message;
        clock_t start;
    };

    std::vector<cv::Mat> ReadFrames(const std::string& path, const std::string& prefix, const std::string& format);
    
    std::array<double, AvgMetricType::AvgMTSize> ApplyMoveDetectionWithTreshold(
        const std::string& fileName,
        const std::vector<cv::Mat>& frames,
        const std::vector<cv::Mat>& markup,
        uchar treshold
    );
    
    std::array<double, AvgMetricType::AvgMTSize> ApplyMoveDetectionWithTresholdGrayBlur(
        const std::string& fileName,
        const std::vector<cv::Mat>& frames,
        const std::vector<cv::Mat>& markup,
        uchar treshold,
        bool blur
    );

    std::array<double, AvgMTSize> ApplyMoveDetectionWithFrameDiffrencing(
        const std::string& fileName,
        const std::vector<cv::Mat>& frames,
        const std::vector<cv::Mat>& markup,
        uchar treshold,
        uint frameCount = 1,
        bool blur = false
    );

    std::array<double, AvgMTSize> ApplyMoveDetectionWithOpticalFlow(
        const std::string& fileName,
        const std::vector<cv::Mat>& frames,
        const std::vector<cv::Mat>& markup,
        uchar treshold
    );
}

