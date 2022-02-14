#include "pch.h"
#include "ai_dnn.h"
#include <fstream>
#include <thread>
#include <chrono>
#include "Inifile.h"

CAIDnn::CAIDnn(string sNamesFile, string sCfgFile, string sWeightsFile, float threshold, string prefix, string directory, SYSTEMTIME _starttime)
	: _prefix(prefix)
	, _directory(directory)
{
	if (_directory.empty() == false)
		_directory += "\\";

	_threshold = threshold;

	_start_time = _starttime;

	CString strNames(sNamesFile.c_str());
	CString strCfg(sCfgFile.c_str());
	CString strWeights(sWeightsFile.c_str());

	if (GetFileAttributes(strNames) != INVALID_FILE_ATTRIBUTES && GetFileAttributes(strCfg) != INVALID_FILE_ATTRIBUTES && GetFileAttributes(strWeights) != INVALID_FILE_ATTRIBUTES) {
		ifstream ifs(sNamesFile);
		string line;
		while (getline(ifs, line))
			_classes.push_back(line);

		_nClasses = _classes.size();

	    _net = readNet(sWeightsFile, sCfgFile);

		if (!_net.empty()) {
#ifdef _GPU
			_net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
			_net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
#else
			_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
			_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
#endif

			_outLayers = _net.getUnconnectedOutLayers();
			_layersNames = _net.getLayerNames();

			_names.resize(_outLayers.size());
			for (int i = 0; i < _outLayers.size(); i++)
				_names[i] = _layersNames[_outLayers[i] - 1];
		}

		_width = 608;
		_height = 608;

		CIniFileW cfgFile;
		if (cfgFile.Load(strCfg.GetBuffer(0))) {
			CString strConfigSession = L"net";

			CString strWidth = cfgFile.GetKeyValue(strConfigSession.GetBuffer(0), L"width").c_str();
			CString strHeight = cfgFile.GetKeyValue(strConfigSession.GetBuffer(0), L"height").c_str();

			if (strWidth.IsEmpty() == false)
				_width = _ttoi(strWidth);
			if (strHeight.IsEmpty() == false)
				_height = _ttoi(strHeight);
		}
	}
}

CAIDnn::~CAIDnn()
{
}

std::vector<bbox2_t> CAIDnn::Postprocess(Mat& frame, const std::vector<Mat>& outs, string windowName, string times)
{
	std::vector<bbox2_t> ret_vects;

	float confThreshold = _threshold, nmsThreshold = 0.4f;

	float prob = 0.0;
	int cx = 0;
	int cy = 0;

	SYSTEMTIME SystemTime, resTime;
	GetLocalTime(&SystemTime);
//	resTime = _start_time - SystemTime;

//	std::difftime(_start_time, SystemTime);

	//char buf[128];
	//sprintf(buf, "%04d%02d%02d%02d%02d%02d%03d", SystemTime.wYear, SystemTime.wMonth, SystemTime.wDay, SystemTime.wHour, SystemTime.wMinute, SystemTime.wSecond, SystemTime.wMilliseconds);
	FILE* fp = nullptr;

//	string strImg = _directory + _prefix + string(buf) + ".png";
//	string strTxt = _directory + _prefix + string(buf) + ".txt";
	string strImg = _directory + _prefix + "-" + times + ".png";
	string strTxt = _directory + _prefix + "-" + times + ".txt";
	imwrite(strImg, frame);

	if (!_net.empty()) {
	
		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<Rect> boxes;
		for (size_t i = 0; i < outs.size(); ++i) {
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold) {
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}

		std::vector<int> indices;
		NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

		bbox2_t vect;
		for (size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];

			if (boxes[idx].x >= frame.cols || boxes[idx].y >= frame.rows)
				continue;

			if (boxes[idx].x < 0)
				boxes[idx].x = 0;

			if (boxes[idx].y < 0)
				boxes[idx].y = 0;

			if ((boxes[idx].x + boxes[idx].width) >= frame.cols)
				boxes[idx].width = frame.cols - boxes[idx].x;

			if ((boxes[idx].y + boxes[idx].height) >= frame.rows)
				boxes[idx].height = frame.rows - boxes[idx].y;

			//if (boxes[idx].width < 10 || boxes[idx].height < 10)
			//	continue;

			memset(&vect, 0x00, sizeof(vect));
			vect.x = boxes[idx].x;
			vect.y = boxes[idx].y;
			vect.w = boxes[idx].width;
			vect.h = boxes[idx].height;
			vect.obj_id = classIds[idx];
			vect.prob = confidences[idx];
			ret_vects.push_back(vect);
		}


#ifdef REMOVEFILE
		if (boxes.size() < 1)
		{
			if (fp)
				fclose(fp);
			return ret_vects;
		}
#endif 

//		string strImg = _directory + _prefix + string(buf) + ".png";
//		string strTxt = _directory + _prefix + string(buf) + ".txt";
//		imwrite(strImg, frame);

		if (indices.size() > 0) {
			fp = fopen(strTxt.c_str(), "wt");
			for (size_t i = 0; i < indices.size(); ++i) 
			{
				int idx = indices[i];

				Rect box = boxes[idx];

				// drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
				if (fp)
					fprintf(fp, "%d %lf %lf %lf %lf\n", classIds[idx], (double)(2 * box.x + box.width) / (2 * frame.cols), (double)(2 * box.y + box.height) / (2 * frame.rows), (double)box.width / frame.cols, (double)box.height / frame.rows);
			}

		}
	}

	if (fp)
		fclose(fp);

/*
	namedWindow(windowName);
	imshow(windowName, frame);
	waitKey(1);
*/

	return ret_vects;
}

void CAIDnn::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

	std::string label;// = format("%.2f", conf);
	if (!_classes.empty()) {
		CV_Assert(classId < (int)_classes.size());
		label = _classes[classId];// +": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseLine);

	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.8, Scalar());
}

std::vector<bbox2_t> CAIDnn::AnalysisProcess(Mat &frame, string windowName, string times)
{
	std::vector<bbox2_t> result_vec;
	Mat blob;
	vector<Mat> outs;

	if (!_net.empty()) {
		blobFromImage(frame, blob, 1 / 255.0, cv::Size(_width, _height), Scalar(0, 0, 0), true, false);
		_net.setInput(blob);

		_net.forward(outs, _names);
	}

	Mat frame2 = frame.clone();
	result_vec = Postprocess(frame2, outs, windowName, times);

	return result_vec;
}
