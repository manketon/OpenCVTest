/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_Opencv_Transform_Tool.cpp
* @brief: ���˵���ļ����ܡ���; (Comment)��
* @author:	minglu2
* @version: 1.0
* @date: 2019/06/11
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
*  <tr> <td>1.0	    <td>2019/06/11	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "CBusin_Opencv_Transform_Tool.h"
#include "Busin_OpenCV_Common_Tool.h"



cv::Mat CBusin_Opencv_Transform_Tool::rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center , double degree, float fScale, const Rect& rect_sub_src, Rect& rect_sub_dst_shrink, const Scalar& borderValue /*= Scalar(0, 0, 0)*/)
{
	degree = -degree;
	double angle = degree  * CV_PI / 180.; // ����
	double a = sin(angle), b = cos(angle);
	int width = src_img_mat.cols;
	int height = src_img_mat.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]

	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6] = {0};
	Mat mat_matrix = Mat(2, 3, CV_32F, map);
	// ��ת����
	//		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 = mat_matrix;
	cv2DRotationMatrix(center, degree, fScale, &map_matrix2);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	Mat img_rotated;

	//��ͼ��������任

	//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�

	//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.

	//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
	cv::warpAffine(src_img_mat, img_rotated, mat_matrix, Size(width_rotate, height_rotate), 1, 0, borderValue);
	//��ȡ�Ӿ�����ԭͼ�еĶ���
	Point points_arr_src[4] = {
		Point(rect_sub_src.x, rect_sub_src.y + rect_sub_src.height)
		, Point(rect_sub_src.x, rect_sub_src.y)
		, Point(rect_sub_src.x + rect_sub_src.width, rect_sub_src.y)
		, Point(rect_sub_src.x + rect_sub_src.width, rect_sub_src.y + rect_sub_src.height)};
	//�Ӿ��η���任��Ķ�������
	Point2f points_arr_dst[4];
	for (int i = 0; i != 4; ++i)
	{
		points_arr_dst[i] = get_dst_point_after_affine(points_arr_src[i], mat_matrix);
		//�Զ����������
		points_arr_dst[i].x =  points_arr_dst[i].x * width_rotate / width;
		points_arr_dst[i].y = points_arr_dst[i].y  * height_rotate / height;
	}
	//�����߶�
// 	CBusin_OpenCV_Common_Tool::instance().draw_lines(img_rotated, points_arr_dst, 4, Scalar(0));
// 	waitKey(0);
	//��ȡ��ת���ζ���
	RotatedRect rRect;
	CBusin_OpenCV_Common_Tool::instance().get_my_RotatedRect(points_arr_dst[0], points_arr_dst[1], points_arr_dst[2], rRect);
	//��������ڽ�upRight����
	if (rRect.size.width < rRect.size.height)
	{
		std::swap(rRect.size.width, rRect.size.height);
		if (rRect.angle < 0)
		{
			rRect.angle += 90;
		}
		else
		{
			rRect.angle -= 90;
		}
	}
	rRect.points(points_arr_dst);
	if (rRect.angle > 90)
	{//����ҵ�
		Point middle_1 = (points_arr_dst[0] + points_arr_dst[3]) / 2;
		Point middle_2 = (points_arr_dst[1] + points_arr_dst[2]) / 2;
		rect_sub_dst_shrink.x = middle_1.x;
		rect_sub_dst_shrink.y = middle_2.y;
		rect_sub_dst_shrink.width = middle_2.x - middle_1.x + 1;
		rect_sub_dst_shrink.height = middle_1.y - middle_2.y + 1;
	}
	else if (fabs(rRect.angle - 90) < 0.0001 || fabs(rRect.angle) < 0.0001)
	{
		rect_sub_dst_shrink = rect_sub_src;
	}
	else if (rRect.angle > 0)
	{
		Point middle_1 = (points_arr_dst[0] + points_arr_dst[3]) / 2;
		Point middle_2 = (points_arr_dst[1] + points_arr_dst[2]) / 2;
		rect_sub_dst_shrink.x = middle_1.x;
		rect_sub_dst_shrink.y = middle_2.y;
		rect_sub_dst_shrink.width = middle_2.x - middle_1.x + 1;
		rect_sub_dst_shrink.height = middle_1.y - middle_2.y + 1;
	}
	else
	{
		Point middle_1 = (points_arr_dst[0] + points_arr_dst[3]) / 2;
		Point middle_2 = (points_arr_dst[1] + points_arr_dst[2]) / 2;
		rect_sub_dst_shrink.x = middle_2.x;
		rect_sub_dst_shrink.y = middle_2.y;
		rect_sub_dst_shrink.width = middle_1.x - middle_2.x + 1;
		rect_sub_dst_shrink.height = middle_1.y - middle_2.y + 1;
	}
	return img_rotated;
}
