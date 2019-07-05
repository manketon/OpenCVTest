/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_Opencv_Transform_Tool.h
* @brief: ���˵���ļ����ܡ���; (Comment)��
* @author:	minglu2
* @version: 1.0
* @date: 2018/10/29
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
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
	* Brief:  ������ָ���㡢���Ⱥͱ���������ת�任
	* Access:    public 
	* Returns:   cv::Mat
	* Qualifier:
	*Parameter: const cv::Mat & src_img_mat -[in]  
	*Parameter: const CvPoint2D32f & center -[in]  ��ת���ĵ�
	*Parameter: double degree -[in]  �Ƕȣ�˳�뷽��Ϊ��ֵ��
	*Parameter: float fScale -[in]  �������ű��� 
	*Parameter: const Scalar& borderValue -[in]  ʹ�ò�ֵ�㷨����չ�߽�ʱ����ɫֵ
	************************************/
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center, double degree, float fScale, const Scalar& borderValue = Scalar(0, 0, 0));
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center
		, double degree, float fScale, const Rect& rect_sub_src, Rect& rect_sub_dst_shrink, const Scalar& borderValue = Scalar(0, 0, 0));
	cv::Point get_dst_point_after_affine(const cv::Point& src_point, const Mat& affine_transform_mat);
	/************************************
	* Method:    get_rotation_matrix_without_loss
	* Brief:  ��ȡ��������ʧ�任����ķ���任�����Լ��������Ĵ�С
	* Access:    public 
	* Returns:   int
	* Qualifier:
	*Parameter: const cv::Mat & src_img_mat -[in]  
	*Parameter: const CvPoint2D32f & center -[in]  ��ת����
	*Parameter: double degree -[in]  �Ƕȣ�˳�뷽��Ϊ��ֵ��
	*Parameter: float fScale -[in]  �������ű���
	*Parameter: Mat& mat_rotation -[out] ����任����
	*Parameter: Size& dSize ����任��Ľ������Ĵ�С
	************************************/
	int get_rotation_matrix_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center
		, double degree, float fScale, Mat& mat_rotation, Size& dSize);

protected:
private:
};
