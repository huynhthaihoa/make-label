#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <fcntl.h>
#include <errno.h>

#include <opencv2/opencv.hpp>

#include "Interface_struct.h"

using namespace std;

using namespace cv;
using namespace dnn;



class CcontainerNumberH {
public:
	int pos_x;
	int pos_y;
	int pos; //라인 선택
	int nMargin = 10;
	int het;
	// int gradient;

	CcontainerNumberH(int x, int y, int hh) :pos_x(x), pos_y(y), het(hh) { this->pos = 1; };

	bool operator<(CcontainerNumberH pc) const {

		if ((this->pos_y + this->het + this->nMargin) > pc.pos_y) pc.pos = 2;
		else pc.pos = 1;

		if (this->pos_x == pc.pos_x)
			return this->pos_y < pc.pos_y;
		else
			return this->pos_x < pc.pos_x;
	}
};

class CAIDnn {
public:
	CAIDnn(string sNamesFile, string sCfgFile, string sWeightsFile, float threshold, string prefix, string directory, SYSTEMTIME starttime);
	~CAIDnn();
	
private:
	dnn::Net _net;
	vector<string> _names;
	vector<int> _outLayers;
	vector<string> _layersNames;
	vector<string> _classes;
	float _threshold = 0.5f;

	string _prefix;
	string _directory;

	//시작시간을 들고옴
	SYSTEMTIME _start_time;
	int _nClasses;

	int _width;
	int _height;

public:
	//std::vector<bbox2_t> AnalysisProcess(Mat &frame, string windowName);
	std::vector<bbox2_t> AnalysisProcess(Mat& frame, string windowName, string times);
private:
	//std::vector<bbox2_t> Postprocess(Mat& frame, const std::vector<Mat>& outs, string windowName);
	std::vector<bbox2_t> Postprocess(Mat& frame, const std::vector<Mat>& outs, string windowName, string times);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
	
};
