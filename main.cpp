#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// OpenCV Version
	cout << "OpenCV Version : " << CV_VERSION << endl;

	// OpenCL�� ����� �� �ִ��� �׽�Ʈ 
	if (!ocl::haveOpenCL()) {
		cout << "���� : OpenCL�� ����� �� ���� �ý����Դϴ�." << endl;
		//return  -1;
	}
	else
	{
		// ���ؽ�Ʈ ����
		ocl::Context context;
		if (!context.create(ocl::Device::TYPE_GPU)) {
			cout << " ���� : ���ؽ�Ʈ�� ������ �� �����ϴ�." << endl;
			return  -1;
		}

		// GPU ��ġ ����
		cout << context.ndevices() << " GPU device (s) detected " << endl;
		for (size_t i = 0; i < context.ndevices(); i++) {
			ocl::Device device = context.device(i);
			cout << " - Device " << i << " --- " << endl;
			cout << " Name : " << device.name() << endl;
			cout << " Availability : " << device.available() << endl;
			cout << "Image Support : " << device.imageSupport() << endl;
			cout << " OpenCL C version : " << device.OpenCL_C_Version() << endl;
		}

		// ��ġ 0 �� ��� 
		ocl::Device(context.device(0));

		// Enable OpenCL
		ocl::setUseOpenCL(true);
	}


	// ���� �ð� ���� 
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
