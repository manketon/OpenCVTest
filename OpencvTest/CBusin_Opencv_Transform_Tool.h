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
	static CBusin_Opencv_Transform_Tool& instance();
	Mat image_rotate(const Mat & src_mat, const cv::Point &center_point, double dDegree, double scale);
	//
	cv::Mat rotate_image__and_shift(const cv::Mat& src_img_mat, const cv::Point &center_point
		, int degree, float fX_shift, float fY_shift);

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
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center, double degree, float fScale, const Scalar& borderValue = Scalar(0, 0, 0));
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center
		, double degree, float fScale, const Rect& rect_sub_src, Rect& rect_sub_dst_shrink, const Scalar& borderValue = Scalar(0, 0, 0));
	cv::Point get_dst_point_after_affine(const cv::Point& src_point, const Mat& affine_transform_mat);
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
		, double degree, float fScale, Mat& mat_rotation, Size& dSize);

protected:
private:
};
