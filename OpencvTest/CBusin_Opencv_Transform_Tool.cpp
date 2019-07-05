/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_Opencv_Transform_Tool.cpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2019/06/11
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/06/11	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "CBusin_Opencv_Transform_Tool.h"
#include "Busin_OpenCV_Common_Tool.h"



CBusin_Opencv_Transform_Tool& CBusin_Opencv_Transform_Tool::instance()
{
	static CBusin_Opencv_Transform_Tool obj;
	return obj;
}

cv::Mat CBusin_Opencv_Transform_Tool::image_rotate(const Mat & src_mat, const cv::Point &center_point, double dDegree, double scale)
{
	CvPoint2D32f center;
	center.x = float(center_point.x);
	center.y = float(center_point.y);

	//计算二维旋转的仿射变换矩阵
	Mat M = cv::getRotationMatrix2D(center, dDegree, scale);

	Mat dst_mat;
	cv::warpAffine(src_mat, dst_mat, M, cvSize(src_mat.cols, src_mat.rows), CV_INTER_LINEAR);
	return dst_mat;
}

cv::Mat CBusin_Opencv_Transform_Tool::rotate_image__and_shift(const cv::Mat& src_img_mat, const cv::Point &center_point , int degree, float fX_shift, float fY_shift)
{
	degree = -degree;
	double angle = degree  * CV_PI / 180.; // 弧度
	double a = sin(angle), b = cos(angle);
	int width = src_img_mat.cols;
	int height = src_img_mat.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b)) + fX_shift * 2;
	int height_rotate = int(width * fabs(a) + height * fabs(b)) + fY_shift * 2;
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]

	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6] = {0};
	Mat map_matrix = Mat(2, 3, CV_32F, map);
	// 旋转中心
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 = map_matrix;
	//这里传入的应该是角度而非弧度，即不需要CV_PI / 180
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);
	map[2] += (width_rotate - width) / 2 + fX_shift;
	map[5] += (height_rotate - height) / 2 + fY_shift;
	Mat img_rotate;

	//对图像做仿射变换

	//CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。

	//如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.

	//CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，
	cv::warpAffine(src_img_mat, img_rotate, map_matrix, Size(width_rotate, height_rotate), INTER_NEAREST, 0, 0);
	return img_rotate;
}

cv::Mat CBusin_Opencv_Transform_Tool::rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center , double degree, float fScale, const Rect& rect_sub_src, Rect& rect_sub_dst_shrink, const Scalar& borderValue /*= Scalar(0, 0, 0)*/)
{
	degree = -degree;
	double angle = degree  * CV_PI / 180.; // 弧度
	double a = sin(angle), b = cos(angle);
	int width = src_img_mat.cols;
	int height = src_img_mat.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]

	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6] = {0};
	Mat mat_matrix = Mat(2, 3, CV_32F, map);
	// 旋转中心
	//		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 = mat_matrix;
	cv2DRotationMatrix(center, degree, fScale, &map_matrix2);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	Mat img_rotated;

	//对图像做仿射变换

	//CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。

	//如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.

	//CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，
	cv::warpAffine(src_img_mat, img_rotated, mat_matrix, Size(width_rotate, height_rotate), 1, 0, borderValue);
	//获取子矩形在原图中的顶点
	Point points_arr_src[4] = {
		Point(rect_sub_src.x, rect_sub_src.y + rect_sub_src.height)
		, Point(rect_sub_src.x, rect_sub_src.y)
		, Point(rect_sub_src.x + rect_sub_src.width, rect_sub_src.y)
		, Point(rect_sub_src.x + rect_sub_src.width, rect_sub_src.y + rect_sub_src.height)};
	//子矩形仿射变换后的顶点坐标
	Point2f points_arr_dst[4];
	for (int i = 0; i != 4; ++i)
	{
		points_arr_dst[i] = get_dst_point_after_affine(points_arr_src[i], mat_matrix);
	}
	//画出线段
// 	CBusin_OpenCV_Common_Tool::instance().draw_lines(img_rotated, points_arr_dst, 4, Scalar(0), "img rotated with Rect rotated");
// 	waitKey(0);
	//获取旋转矩形对象
	RotatedRect rRect;
	CBusin_OpenCV_Common_Tool::instance().get_my_RotatedRect(points_arr_dst[0], points_arr_dst[1], points_arr_dst[2], rRect);
	//在旋转所得子矩形中求最大内接upRight矩形
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
	//找到子矩形的最大内接upRight矩形
	//TODO::下面算法待优化
	if (rRect.angle > 90)
	{//左高右低
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

cv::Mat CBusin_Opencv_Transform_Tool::rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center, double degree, float fScale, const Scalar& borderValue /*= Scalar(0, 0, 0)*/)
{
	degree = -degree;
	double angle = degree  * CV_PI / 180.; // 弧度
	double a = sin(angle), b = cos(angle);
	int width = src_img_mat.cols;
	int height = src_img_mat.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]

	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6] = {0};
	Mat map_matrix = Mat(2, 3, CV_32F, map);
	// 旋转中心
	//		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 = map_matrix;
	cv2DRotationMatrix(center, degree, fScale, &map_matrix2);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	Mat img_rotated;

	//对图像做仿射变换

	//CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。

	//如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.

	//CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，
	cv::warpAffine(src_img_mat, img_rotated, map_matrix, Size(width_rotate, height_rotate), 1, 0, borderValue);
	return img_rotated;
}

cv::Point CBusin_Opencv_Transform_Tool::get_dst_point_after_affine(const cv::Point& src_point, const Mat& affine_transform_mat)
{
	cv::Point dst_point;
	if (affine_transform_mat.type() == CV_32FC1)
	{
		dst_point.x = cvRound(affine_transform_mat.at<float>(0, 0) * src_point.x + affine_transform_mat.at<float>(0, 1) * src_point.y + affine_transform_mat.at<float>(0, 2));
		dst_point.y = cvRound(affine_transform_mat.at<float>(1, 0) * src_point.x + affine_transform_mat.at<float>(1, 1) * src_point.y + affine_transform_mat.at<float>(1, 2));
	}
	else if (affine_transform_mat.type() == CV_64FC1)
	{
		dst_point.x = cvRound(affine_transform_mat.at<double>(0, 0) * src_point.x + affine_transform_mat.at<double>(0, 1) * src_point.y + affine_transform_mat.at<double>(0, 2));
		dst_point.y = cvRound(affine_transform_mat.at<double>(1, 0) * src_point.x + affine_transform_mat.at<double>(1, 1) * src_point.y + affine_transform_mat.at<double>(1, 2));
	}
	else
	{
		printf("func:%s | error\n", __FUNCTION__);
		exit(-1);
	}
	return dst_point;
}

int CBusin_Opencv_Transform_Tool::get_rotation_matrix_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center , double degree, float fScale, Mat& mat_rotation, Size& dSize)
{
	degree = -degree;
	double angle = degree  * CV_PI / 180.; // 弧度
	double a = sin(angle), b = cos(angle);
	int width = src_img_mat.cols;
	int height = src_img_mat.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6] = {0};
	Mat map_matrix = Mat(2, 3, CV_32F, map);
	CvMat map_matrix2 = map_matrix;
	cv2DRotationMatrix(center, degree, fScale, &map_matrix2);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	dSize = Size(width_rotate, height_rotate);
	mat_rotation = map_matrix.clone();
	return 0;
}
