/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_Opencv_Transform_Tool.h
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2018/10/29
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2018/10/29	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
using namespace cv;
class CBusin_Opencv_Transform_Tool
{
public:
	static CBusin_Opencv_Transform_Tool& instance()
	{
		static CBusin_Opencv_Transform_Tool obj;
		return obj;
	}
	Mat image_rotate(const Mat & src_mat, const cv::Point &center_point, double dDegree, double scale)
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
	//
	cv::Mat rotate_image__and_shift(const cv::Mat& src_img_mat, const cv::Point &center_point
		, int degree, float fX_shift, float fY_shift)
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

	/************************************
	* Method:    rotate_image_without_loss
	* Brief:  将矩阵按指定点、弧度和比例进行旋转变换
	* Access:    public 
	* Returns:   cv::Mat
	* Qualifier:
	*Parameter: const cv::Mat & src_img_mat -[in]  
	*Parameter: const CvPoint2D32f & center -[in]  旋转中心点
	*Parameter: double degree -[in]  角度（顺针方向为正值）
	*Parameter: float fScale -[in]  整体缩放比例 
	*Parameter: const Scalar& borderValue -[in]  使用插值算法来扩展边界时的颜色值
	************************************/
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center, double degree, float fScale, const Scalar& borderValue = Scalar(0, 0, 0))
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
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center
		, double degree, float fScale, Mat& mat_matrix, const Scalar& borderValue = Scalar(0, 0, 0))
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
		mat_matrix = Mat(2, 3, CV_32F, map);
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
		return img_rotated;
	}
	/************************************
	* Method:    get_rotation_matrix_without_loss
	* Brief:  获取进行无损失变换所需的仿射变换矩阵以及结果矩阵的大小
	* Access:    public 
	* Returns:   int
	* Qualifier:
	*Parameter: const cv::Mat & src_img_mat -[in]  
	*Parameter: const CvPoint2D32f & center -[in]  旋转中心
	*Parameter: double degree -[in]  角度（顺针方向为正值）
	*Parameter: float fScale -[in]  整体缩放比例
	*Parameter: Mat& mat_rotation -[out] 仿射变换矩阵
	*Parameter: Size& dSize 无损变换后的结果矩阵的大小
	************************************/
	int get_rotation_matrix_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center
		, double degree, float fScale, Mat& mat_rotation, Size& dSize)
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

protected:
private:
};
