/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_Opencv_Contour.cpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2019/07/05
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/07/05	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "Busin_Opencv_Contour.h"

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

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
	namedWindow( "Contours", CV_WINDOW_NORMAL );
	imshow( "Contours", drawing );
	imwrite("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_result/result.jpg", drawing);
}


CBusin_OpenCV_Contour_Tool& CBusin_OpenCV_Contour_Tool::instance()
{
	static CBusin_OpenCV_Contour_Tool obj;
	return obj;
}

int CBusin_OpenCV_Contour_Tool::test(const string& str_img_file_path)
{
	/// 加载源图像
	src = imread(str_img_file_path.c_str(), 1);
	if (src.empty())
	{
		std::cout << __FUNCTION__ << " | fail to open file:" << str_img_file_path << endl;
		return -1;
	}
	/// 转成灰度并模糊化降噪
	cvtColor( src, src_gray, CV_BGR2GRAY );
	//		blur( src_gray, src_gray, Size(3,3) );

	/// 创建窗体
	char* source_window = "Source";
	namedWindow( source_window, CV_WINDOW_NORMAL );
	imshow( source_window, src );

	createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
	thresh_callback( 0, 0 );

	waitKey(0);
	return(0);
}
