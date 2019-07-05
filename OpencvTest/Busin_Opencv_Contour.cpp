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
