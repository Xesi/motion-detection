#include "utils.h"

#include <algorithm>
#include <assert.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <random>

using namespace std;
using namespace cv;
using namespace MoveDetection;

int main(int argc, char const *argv[]) {

    string method = "method4";
    string pathToFolder = "data";
    vector<string> sequenceNames = {"I_BS_01", "I_IL_01", "I_IL_02"};

    if (argc < 2) {
        std::cerr << "It is necessary to choose a detection method. (e.g. methodN, N = 1,2,3)" << std::endl;
        //return -1;
    } else {
        method = argv[1];
    }

    if (argc >= 3) {
        pathToFolder = argv[2];
    }
    for(uint i = 3; i < argc; ++i) {
        sequenceNames.push_back(argv[i]);
    }

    Timer timer("Total");

    vector<vector<Mat>> frames(sequenceNames.size());
    vector<vector<Mat>> markup(sequenceNames.size());

    for(uint i = 0; i < sequenceNames.size(); ++i) {
        Timer timer("Reading dataset " + sequenceNames[i]);
        frames[i] = ReadFrames(pathToFolder + "/" + sequenceNames[i] + "/" + sequenceNames[i] + "/", sequenceNames[i] + "-", ".bmp");
        markup[i] = ReadFrames(pathToFolder + "/" + sequenceNames[i] + "/" + sequenceNames[i] + "-GT/", sequenceNames[i] + "-GT_", ".png");
        
        if (frames[i].empty()) {
            std::cerr << "Frames was not found\n";
            return -1;
        } else if (frames[i].size() != markup[i].size()) {
            std::cerr << "The number of images and markup found is different\n";
            return -1;
        } else {
            std::cerr << frames[i].size() << " frames were found." << std::endl;
        }
    }

    uchar bestTreshold = 0;
    double bestValue = 0;
    uint bestIndex = 0;
    ofstream out("result_" + method + ".txt");

    for(uint treshold = 0; treshold < 255; treshold += 1) {

        vector<array<double, AvgMTSize>> metrics(sequenceNames.size());

        double current = 0;
        for(uint i = 0; i < sequenceNames.size(); ++i) {
            if (method == "method1") {
                metrics[i] = ApplyMoveDetectionWithTreshold(sequenceNames[i], frames[i], markup[i], treshold);
            } else if (method == "method2") {
                metrics[i] = ApplyMoveDetectionWithTresholdGrayBlur(sequenceNames[i], frames[i], markup[i], treshold, true);
            } else if (method == "method3") {
                metrics[i] = ApplyMoveDetectionWithFrameDiffrencing(sequenceNames[i], frames[i], markup[i], treshold, 1, true);
            } else if (method == "method4") {
                metrics[i] = ApplyMoveDetectionWithOpticalFlow(sequenceNames[i], frames[i], markup[i], treshold);
            } else {
                throw runtime_error("unknown method \"" + method + "\".");
            }
            current += metrics[i][AvgTPR] - metrics[i][AvgFPR];

            out << sequenceNames[i] << "," << int(treshold) << "," << metrics[i][AvgAcc] << ","
                << metrics[i][AvgTPR] << "," << metrics[i][AvgFPR] << endl;
        }

        if (current > bestValue) {
            bestValue = current;
            bestTreshold = treshold;
        }
    }
    out.close();

    cerr << "bestTreshold = " << int(bestTreshold) << endl;
    cerr << "bestValue = " << bestValue << endl;
    cerr << endl;

    for(uint i = 0; i < sequenceNames.size(); ++i) {
        cerr << "best treshold:" << endl;
        array<double, AvgMTSize> bestMetrics;
        if (method == "method1") {
            bestMetrics = ApplyMoveDetectionWithTreshold(sequenceNames[i], frames[i], markup[i], bestTreshold);
        } else if (method == "method2") {
            bestMetrics = ApplyMoveDetectionWithTresholdGrayBlur(sequenceNames[i], frames[i], markup[i], bestTreshold, true);
        } else if (method == "method3") {
            bestMetrics = ApplyMoveDetectionWithFrameDiffrencing(sequenceNames[i], frames[i], markup[i], bestTreshold, 1, true);
        } else if (method == "method4") {
            bestMetrics = ApplyMoveDetectionWithOpticalFlow(sequenceNames[i], frames[i], markup[i], bestTreshold);
        } 
    }

    // waitKey(0);
    return 0;
}


