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

		//�����ά��ת�ķ���任����
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
		double angle = degree  * CV_PI / 180.; // ����
		double a = sin(angle), b = cos(angle);
		int width = src_img_mat.cols;
		int height = src_img_mat.rows;
		int width_rotate = int(height * fabs(a) + width * fabs(b)) + fX_shift * 2;
		int height_rotate = int(width * fabs(a) + height * fabs(b)) + fY_shift * 2;
		//��ת����map
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]

		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float map[6] = {0};
		Mat map_matrix = Mat(2, 3, CV_32F, map);
		// ��ת����
		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
		CvMat map_matrix2 = map_matrix;
		//���ﴫ���Ӧ���ǽǶȶ��ǻ��ȣ�������ҪCV_PI / 180
		cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);
		map[2] += (width_rotate - width) / 2 + fX_shift;
		map[5] += (height_rotate - height) / 2 + fY_shift;
		Mat img_rotate;

		//��ͼ��������任

		//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�

		//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.

		//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
		cv::warpAffine(src_img_mat, img_rotate, map_matrix, Size(width_rotate, height_rotate), INTER_NEAREST, 0, 0);
		return img_rotate;
	}

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
	************************************/
	cv::Mat rotate_image_without_loss(const cv::Mat& src_img_mat, const CvPoint2D32f& center, double degree, float fScale)
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
		Mat map_matrix = Mat(2, 3, CV_32F, map);
		// ��ת����
//		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
		CvMat map_matrix2 = map_matrix;
		cv2DRotationMatrix(center, degree, fScale, &map_matrix2);
		map[2] += (width_rotate - width) / 2;
		map[5] += (height_rotate - height) / 2;
		Mat img_rotated;

		//��ͼ��������任

		//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�

		//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.

		//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
		cv::warpAffine(src_img_mat, img_rotated, map_matrix, Size(width_rotate, height_rotate), 1, 0, 0);
		return img_rotated;
	}

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
		, double degree, float fScale, Mat& mat_rotation, Size& dSize)
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
