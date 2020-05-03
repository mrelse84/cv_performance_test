#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// OpenCV Version
	cout << "OpenCV Version : " << CV_VERSION << endl;

	// OpenCL을 사용할 수 있는지 테스트 
	if (!ocl::haveOpenCL()) {
		cout << "에러 : OpenCL을 사용할 수 없는 시스템입니다." << endl;
		//return  -1;
	}
	else
	{
		// 컨텍스트 생성
		ocl::Context context;
		if (!context.create(ocl::Device::TYPE_GPU)) {
			cout << " 에러 : 컨텍스트를 생성할 수 없습니다." << endl;
			return  -1;
		}

		// GPU 장치 정보
		cout << context.ndevices() << " GPU device (s) detected " << endl;
		for (size_t i = 0; i < context.ndevices(); i++) {
			ocl::Device device = context.device(i);
			cout << " - Device " << i << " --- " << endl;
			cout << " Name : " << device.name() << endl;
			cout << " Availability : " << device.available() << endl;
			cout << "Image Support : " << device.imageSupport() << endl;
			cout << " OpenCL C version : " << device.OpenCL_C_Version() << endl;
		}

		// 장치 0 번 사용 
		ocl::Device(context.device(0));

		// Enable OpenCL
		ocl::setUseOpenCL(true);
	}


	// 실행 시간 측정 
	static int64 start, end;
	static float time;

	// Load Lena Image
	Mat img = imread("C:/images/opencv_images/lena.jpg");
	imshow("image", img);
	waitKey();

	// Convert to Gray
	//=======================================================================================
	Mat img_gray;
	start = getTickCount();
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	end = getTickCount();
	time = (end - start) / getTickFrequency() * 1000;
	cout << "cvtColor - Processing Time : " << time << " msec. " << endl;

	// Threshold
	//=======================================================================================
	Mat img_th;
	start = getTickCount();
	threshold(img_gray, img_th, 127, 255, THRESH_BINARY);
	end = getTickCount();
	time = (end - start) / getTickFrequency() * 1000;
	cout << "threshold - Processing Time : " << time << " msec. " << endl;

	// Canny Edge
	//=======================================================================================
	Mat img_canny;
	start = getTickCount();
	Canny(img_gray, img_canny, 100, 127);
	end = getTickCount();
	time = (end - start) / getTickFrequency() * 1000;
	cout << "Canny - Processing Time : " << time << " msec. " << endl;

	imshow("image_gray", img_gray);
	imshow("image_canny", img_canny);
	waitKey();

	return 0;
}
