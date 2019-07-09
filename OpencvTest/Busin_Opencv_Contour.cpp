/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_Opencv_Contour.cpp
* @brief: ���˵���ļ����ܡ���; (Comment)��
* @author:	minglu2
* @version: 1.0
* @date: 2019/07/05
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
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
	//Ϊ�˸��õ�׼ȷ�ԣ�ʹ�ö�����ͼ�� ��ˣ����ҵ�����֮ǰ��Ӧ����ֵ��canny��Ե��⡣
	/// ��Canny���Ӽ���Ե��Canny�õ��ı���Ϊ��ɫ
	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	/// Ѱ������
	//��OpenCV�У��ҵ���������Ӻ�ɫ�������ҵ���ɫ���塣�������ס��Ҫ�ҵ��Ķ���Ӧ���ǰ�ɫ������Ӧ���Ǻ�ɫ��
	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// �������
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	}

	/// �ڴ�������ʾ���
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
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
	/// ����Դͼ��
	src = imread(str_img_file_path.c_str(), 1);

	/// ת�ɻҶȲ�ģ��������
	cvtColor( src, src_gray, CV_BGR2GRAY );
	//		blur( src_gray, src_gray, Size(3,3) );

	/// ��������
	char* source_window = "Source";
	namedWindow( source_window, CV_WINDOW_AUTOSIZE );
	imshow( source_window, src );

	createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
	thresh_callback( 0, 0 );

	waitKey(0);
	return(0);
}
