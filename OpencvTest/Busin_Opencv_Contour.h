#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );


/** @function thresh_callback */
void thresh_callback(int, void* )
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//为了更好的准确性，使用二进制图像 因此，在找到轮廓之前，应用阈值或canny边缘检测。
	/// 用Canny算子检测边缘，Canny得到的背景为黑色
	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	/// 寻找轮廓
	//在OpenCV中，找到轮廓就像从黑色背景中找到白色物体。所以请记住，要找到的对象应该是白色，背景应该是黑色。
	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// 绘出轮廓
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	}

	/// 在窗体中显示结果
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", drawing );
	imwrite("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_result/result.jpg", drawing);
}

class CBusin_OpenCV_Contour_Tool
{
public:
	static CBusin_OpenCV_Contour_Tool& instance()
	{
		static CBusin_OpenCV_Contour_Tool obj;
		return obj;
	}
	/** @function main */
	int test(const string& str_img_file_path)
	{
		/// 加载源图像
		src = imread(str_img_file_path.c_str(), 1);

		/// 转成灰度并模糊化降噪
		cvtColor( src, src_gray, CV_BGR2GRAY );
//		blur( src_gray, src_gray, Size(3,3) );

		/// 创建窗体
		char* source_window = "Source";
		namedWindow( source_window, CV_WINDOW_AUTOSIZE );
		imshow( source_window, src );

		createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
		thresh_callback( 0, 0 );

		waitKey(0);
		return(0);
	}

protected:
private:
};
