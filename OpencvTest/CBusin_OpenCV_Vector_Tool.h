/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_OpenCV_Vector_Tool.h
* @brief: �������㣬float��ɴ���һ������
* @author:	minglu2
* @version: 1.0
* @date: 2018/08/27
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
*  <tr> <td>1.0	    <td>2018/08/27	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once


#ifdef __cplusplus  
extern "C" {  
//����C���Խӿڡ������ͷ�ļ�
#endif  
#ifdef __cplusplus  
}  
#endif  
//����C++ͷ�ļ������Ǳ�׼��ͷ�ļ���������Ŀͷ�ļ�
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
//�궨��

//���Ͷ���
class  CBusin_OpenCV_Vector_Tool
{
public:
	typedef cv::Point2f Type_Vector;
	static CBusin_OpenCV_Vector_Tool& instance();
	//�������ȡ��Ӧ������,����AB
	Type_Vector get_vector_from_points(const Point& point_A, const Point& point_B);
	//����ֱ��������ABC�����ݵ�A�͵�C�Լ���B.x��B.y
	int get_PointB_of_triangle_ABC(const Point& point_A, const Point& point_C, Point& point_B);
	/************************************
	* Method:    is_special_right_triangle
	* Brief:  �ж����Ƿ��ܹ���ɱ߳���Ϊ3:4:5��ֱ�������Σ����ҽ�AΪ36.87�ȣ���BΪ53.13�ȣ���CΪֱ��
	* Access:    public 
	* Returns:   bool
	* Qualifier:
	*Parameter: const cv::Point & point_A -[in/out]  
	*Parameter: const cv::Point & point_B -[in/out]  
	*Parameter: const cv::Point & point_C -[in/out]  
	************************************/
	bool is_special_right_triangle(const cv::Point& point_A, const cv::Point& point_B, const cv::Point& point_C);
	double get_euclidean(const cv::Point& pointO, const cv::Point& pointA);
	int get_pointB_for_right_triangle(const cv::Point& point_A, const cv::Point& point_B
		, const cv::Point& point_C, cv::Point& dst_pointB);

	int get_pointB_for_right_triangle(const cv::Point& point_A, const cv::Point& point_B
		, const cv::Point& point_C, const double& dDistance_BC, cv::Point& dst_pointB);
	int test_find_PointB();
protected:
private:
};
//����ԭ�Ͷ���
