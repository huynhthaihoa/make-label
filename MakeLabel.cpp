// MakeLabel.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "pch.h"
#include "framework.h"
#include "MakeLabel.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <fstream>

#include <math.h>

#include "inifile.h"
#include "ai_dnn.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>


#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "swscale.lib")
}

#pragma warning(disable:4018)
#pragma warning(disable:4819)
#pragma warning(disable:4996)

#include <libavutil/log.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#ifndef _DEBUG
//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#endif

// 유일한 애플리케이션 개체입니다.



CWinApp theApp;

using namespace std;

int s_nStep = 30;

mutex _mtx;
condition_variable _cond;

thread _thr_ai;

bool _bLoop = false;
std::pair<cv::Mat,std::string> q;
queue<pair<Mat,std::string>> _Frames;

CAIDnn* g_pDetect = nullptr;

void AIThread();
void pushFrame(Mat& frame,std::string times);
void waitForFinish();

//string format 추가 
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args);


int main(int argc, char* argv[])
{
    av_log_set_level(AV_LOG_QUIET);

    if (argc != 2)
        return -1;

    std::string movfile = argv[1];

    TCHAR szPath[MAX_PATH];
    ::GetModuleFileName(NULL, szPath, MAX_PATH);

    CString strPath = szPath;
    if (0 < strPath.ReverseFind('\\')) {
        strPath = strPath.Left(strPath.ReverseFind('\\'));
    }

    ::SetCurrentDirectory(strPath);

    string sPrefix;
    string sDirectory;

    CIniFileW  iniFile;
    CString iniFileName = L"CFG.ini";
    if (iniFile.Load(iniFileName.GetBuffer(0))) {
        CString strConfigSession = L"Config";

        sPrefix = movfile.substr(movfile.find_last_of("\\") + 1);
        sPrefix = sPrefix.substr(0, sPrefix.find_last_of("."));// +"-";

        //CString strPrefix = iniFile.GetKeyValue(strConfigSession.GetBuffer(0), L"Prefix").c_str();
        //if (strPrefix.IsEmpty() == false)
        //    sPrefix = CT2CA(strPrefix);

        //CString strDirectory = iniFile.GetKeyValue(strConfigSession.GetBuffer(0), L"Directory").c_str();
        //if (strDirectory.IsEmpty() == false)
        //    sDirectory = CT2CA(strDirectory);

        sDirectory = "E:\\kisa_out\\" + sPrefix;
        if (cv::utils::fs::exists(sDirectory) == false)
            cv::utils::fs::createDirectory(sDirectory);


        CString strStep = iniFile.GetKeyValue(strConfigSession.GetBuffer(0), L"Step").c_str();
        if (strStep.IsEmpty() == false)
            s_nStep = _ttoi(strStep);
    }

    if (s_nStep <= 0)
        s_nStep = 30;

    av_register_all();
    AVFormatContext* pFormatCtx = avformat_alloc_context();

    if (avformat_open_input(&pFormatCtx, movfile.c_str(), NULL, NULL) != 0) {
        cout << "Couldn't open input stream." << endl;
        return -1;
    }

    // Get video file information
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {
        cout << "Couldn't find stream information." << endl;
        return -1;
    }

    int videoindex = -1;
    for (int i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoindex = i;
            break;
        }
    }

    if (videoindex == -1) {
        cout << "Didn't find a video stream." << endl;
        return -1;
    }

    AVCodecContext* pCodecCtx = pFormatCtx->streams[videoindex]->codec;
    AVCodec* pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
    if (pCodec == NULL) {
        cout << "Codec not found." << endl;
        return -1;
    }

    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
        cout << "Could not open codec." << endl;
        return -1;
    }

    _bLoop = true;
    _thr_ai = thread(AIThread);


    //시작시간을 들고옴
    SYSTEMTIME gStart_SystemTime;
    GetLocalTime(&gStart_SystemTime);

    g_pDetect = new CAIDnn("obj.names", "obj.cfg", "obj.weights", 0.5f, sPrefix, sDirectory, gStart_SystemTime);
    if (!g_pDetect) {
        avcodec_close(pCodecCtx);
        avformat_close_input(&pFormatCtx);
        return -1;
    }

    AVFrame* pFrame = av_frame_alloc();

    AVFrame dst;
    cv::Mat frame = cv::Mat(pCodecCtx->height, pCodecCtx->width, CV_8UC3);
    dst.data[0] = (uint8_t*)frame.data;
    dst.linesize[0] = 3 * pCodecCtx->width;
    avpicture_fill((AVPicture*)&dst, dst.data[0], AVPixelFormat::AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);

    AVPacket* packet = (AVPacket*)av_malloc(sizeof(AVPacket));

    // debug output file information
 // cout << "--------------- File Information ----------------" << endl;;
    av_dump_format(pFormatCtx, 0, movfile.c_str(), 0);
    //  cout << "-------------------------------------------------" << endl;

        // to cut just above the width, in order to better display
    struct SwsContext* img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);


    int VSI = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pCodec, 0);
    while (av_read_frame(pFormatCtx, packet) >= 0)
    {


        if (packet->stream_index == videoindex)// read a compressed data
        {
            std::string times;
            int success = 0;
            int ret = avcodec_decode_video2(pCodecCtx, pFrame, &success, packet);// decode a compressed data
            {

                double totalScond = av_q2d(pFormatCtx->streams[VSI]->time_base) * packet->pts; //현재 시간
                double integer, MiliSecond;
                MiliSecond = modf(totalScond, &integer);

                int resMin = totalScond / 60;
                int resSecond = int(totalScond) % 60;

                //int result = (int)resSecond;

                
                times = string_format("%02dm%02ds%03dms", resMin, resSecond, int(MiliSecond * 1000));
                std::cout << times << std::endl;
                //std::cout << resMin << "분" << resSecond << "초 " << std::endl;
                int k = 0;
                
            }
                
            if (ret < 0)
            {
                cout << "Decode Error." << endl;
                break;
            }
            if (success)
            {
                sws_scale(img_convert_ctx, (const uint8_t* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height, dst.data, dst.linesize);

                pushFrame(frame, times);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        av_free_packet(packet);
    }

    waitForFinish();

    if (_bLoop) {
        _bLoop = false;
        if (_thr_ai.joinable())
            _thr_ai.join();
    }

    if (g_pDetect) {
        delete g_pDetect;
        g_pDetect = nullptr;
    }

    sws_freeContext(img_convert_ctx);
    // close the file and release memory
    av_frame_free(&pFrame);
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);

    return 0;
}

std::vector<std::string> objects_names_from_file(std::string const filename)
{
    std::vector<std::string> file_lines;

    std::ifstream file(filename);

    if (!file.is_open()) return file_lines;

    for (std::string line; getline(file, line);) file_lines.push_back(line);

    return file_lines;
}

void AIThread()
{
    Mat blob;
    vector<Mat> outs;

    int count = 0;
    while (_bLoop) {
        unique_lock<mutex> lock(_mtx);
        if (_Frames.size() <= 0)
            _cond.wait(lock);

        if (_Frames.size() <= 0)
            break;

        std::pair<cv::Mat,std::string> q = _Frames.front();
        Mat frame = q.first;
        std::string _time = q.second;
        _Frames.pop();
        lock.unlock();

        if (count == 0) {
            if (g_pDetect)
                std::vector<bbox2_t> detect_vec = g_pDetect->AnalysisProcess(frame, "detect", _time);
        }

        if (s_nStep > 1)
            count = (count + 1) % s_nStep;
    }
}

//void pushFrame(Mat& frame)
void pushFrame(Mat& frame, std::string times)
{
    bool bNotify = false;
    unique_lock<mutex> lock(_mtx);

    int count = _Frames.size();
    if (count <= 0)
        bNotify = true;
    std::pair<cv::Mat, std::string> p;
    p.first = frame.clone();
    p.second = times;
    _Frames.push(p);

    if (bNotify)
        _cond.notify_one();
}

void waitForFinish()
{
    while (1) {
        unique_lock<mutex> lock(_mtx);
        if (_Frames.size() <= 0) {
            break;
        }
        lock.unlock();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    _cond.notify_one();
}

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0' 
    if (size <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside }
}