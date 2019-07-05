/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_OpenCV_Vector_Tool.cpp
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
#include "CBusin_OpenCV_Vector_Tool.h"



CBusin_OpenCV_Vector_Tool& CBusin_OpenCV_Vector_Tool::instance()
{
	static CBusin_OpenCV_Vector_Tool obj;
	return obj;
}

CBusin_OpenCV_Vector_Tool::Type_Vector CBusin_OpenCV_Vector_Tool::get_vector_from_points(const Point& point_A, const Point& point_B)
{
	Type_Vector result_vec;
	result_vec.x = point_B.x - point_A.x;
	result_vec.y = point_B.y - point_A.y;
	return result_vec;
}

int CBusin_OpenCV_Vector_Tool::get_PointB_of_triangle_ABC(const Point& point_A, const Point& point_C, Point& point_B)
{
	//����CA*����CBΪ0���õ���B��x��y����Ĺ�ϵ
	point_B.y = ((point_A.y - point_C.y) * point_C.y + (point_A.x - point_C.x)*point_C.x + (point_C.x - point_A.x) * point_B.x)
		* 1.0 / (point_A.y - point_C.y);
	return 0;
}

bool CBusin_OpenCV_Vector_Tool::is_special_right_triangle(const cv::Point& point_A, const cv::Point& point_B, const cv::Point& point_C)
{
	double dDistance_AB = get_euclidean(point_A, point_B);
	double dDisstance_BC = get_euclidean(point_B, point_C);
	double dDistance_AC = get_euclidean(point_A, point_C);
	//����֮�ʹ��ڵ�����
	if (dDisstance_BC + dDistance_AC <= dDistance_AB)
	{
		printf("func:%s | AC + BC <= AB.\n", __FUNCTION__);
		return false;
	}
	//�߳�������Ӧ��Ϊ3:4:5
	double dRatio_AC_AB = dDistance_AC / dDistance_AB;
	double dRatio_BC_AB = dDisstance_BC / dDistance_AB;
	//�нǶ���
	double dDegree_BAC =  acos(dDistance_AC / dDistance_AB) * 180 /CV_PI;
	double dDegree_ABC = acos(dDisstance_BC / dDistance_AB) * 180 /CV_PI;
#ifdef _DEBUG_MingLu
	std::cout << "func:" << __FUNCTION__ << ", point A:" << point_A << ", point B:" << point_B << ", point C:" << point_C << std::endl;
	std::cout << "\t" <<"dDistance_AB:" << dDistance_AB << ", dDisstance_BC:" << dDisstance_BC << ", dDistance_AC:" << dDistance_AC << std::endl;
	std::cout << "\t" << "dRatio_AC_AB:" << dRatio_AC_AB << ",dRatio_BC_AB:" << dRatio_BC_AB << std::endl;
	std::cout << "\t" << "degree of BAC:" << dDegree_BAC << ", degree of ABC:" << dDegree_ABC << std::endl;
#endif
	// 	double dBias_edge = 0.1;
	// 	if (fabs(dRatio_AC_AB - 0.8 ) < dBias_edge && fabs(dRatio_BC_AB - 0.6 ) < dBias_edge)
	// 	{//�߳������ϵ
	// 		return true;
	// 	}
	double dBias_degree = 0.3;
	if (fabs(dDegree_BAC - 36.87) < dBias_degree && fabs(dDegree_ABC - 53.13) < dBias_degree)
	{//�Ƕ������ϵ
		std::cout << "func:" << __FUNCTION__ << ", point A:" << point_A << ", point B:" << point_B << ", point C:" << point_C << std::endl;
		std::cout << "\t" <<"dDistance_AB:" << dDistance_AB << ", dDisstance_BC:" << dDisstance_BC << ", dDistance_AC:" << dDistance_AC << std::endl;
		std::cout << "\t" << "dRatio_AC_AB:" << dRatio_AC_AB << ",dRatio_BC_AB:" << dRatio_BC_AB << std::endl;
		std::cout << "\t" << "degree of BAC:" << dDegree_BAC << ", degree of ABC:" << dDegree_ABC << std::endl;
		return true;
	}

	//		printf("func:%s | AC/AB:%f, BC/AB:%f\n", __FUNCTION__, dRatio_AC_AB, dRatio_BC_AB);
	return false;
}

double CBusin_OpenCV_Vector_Tool::get_euclidean(const cv::Point& pointO, const cv::Point& pointA)
{
	double distance;  
	distance = powf((pointO.x - pointA.x),2) + powf((pointO.y - pointA.y),2);  
	distance = sqrtf(distance);  
	return distance;
}

int CBusin_OpenCV_Vector_Tool::get_pointB_for_right_triangle(const cv::Point& point_A, const cv::Point& point_B , const cv::Point& point_C, const double& dDistance_BC, cv::Point& dst_pointB)
{
	int nOperator_flag = 0;
	if (point_B.x >= point_C.x)
	{//B����C���Ҳ�
		nOperator_flag = 1;
	}
	else
	{//B����C�����
		nOperator_flag = -1;
	}

	for ( int  x  =  0; nOperator_flag * x <= (dDistance_BC + 10); x += nOperator_flag)
	{//��һ����Χ������
		dst_pointB = Point(point_C.x + x, 0);
		get_PointB_of_triangle_ABC(point_A, point_C, dst_pointB);
		if (is_special_right_triangle(point_A, dst_pointB, point_C))
		{
			std::cout << "func:" << __FUNCTION__ << ", find" << std::endl;
			return 0;
		}
	}
	std::cout << "func:" << __FUNCTION__ << ", not find" << std::endl;
	return 10109;
}

int CBusin_OpenCV_Vector_Tool::get_pointB_for_right_triangle(const cv::Point& point_A, const cv::Point& point_B , const cv::Point& point_C, cv::Point& dst_pointB)
{
	double dDistance_BC = get_euclidean(point_B, point_C);
	for ( int  x = -dDistance_BC/3; x <= dDistance_BC/3; ++x)
	{//��ԭ��B��ĸ�������Ŀ���B����ʹ�õ�A����C�Լ�Ŀ���B��ɵ�������Ϊ����ֱ��������
		dst_pointB = Point(point_B.x + x, 0);
		get_PointB_of_triangle_ABC(point_A, point_C, dst_pointB);
		if (is_special_right_triangle(point_A, dst_pointB, point_C))
		{
			std::cout << "func:" << __FUNCTION__ << ", find" << std::endl;
			return 0;
		}
	}
	std::cout << "func:" << __FUNCTION__ << ", not find" << std::endl;
	return 10109;
}

int CBusin_OpenCV_Vector_Tool::test_find_PointB()
{
	const cv::Point point_A(1106, 412);
	const cv::Point point_B(1255, 551);
	const cv::Point point_C(1141, 575);
	double dDistance_AB = 203.769;
	double dLen_AB = 23;
	double dLen_BC = 13.8;
	double dScale = dLen_AB / dDistance_AB;
	//��BC��ͼƬ�еĳ��ȣ�Ϊ�˹̶�x�ķ�Χ
	double dDistance_BC = dLen_BC / dScale;
	cv::Point dst_pointB;
	int ret = 0;
	ret = get_pointB_for_right_triangle(point_A, point_B, point_C, dDistance_BC, dst_pointB);
	//		ret = get_pointB_for_right_triangle(point_A, point_B, point_C, dst_pointB);
	if (0 == ret)
	{
		std::cout << "fun:" << __FUNCTION__ << ", dst pointB:" << dst_pointB << std::endl;
	}
	else
	{
		std::cerr << "func:" <<__FUNCTION__ << " failed to find" << std::endl;
	}
	return ret;
}
