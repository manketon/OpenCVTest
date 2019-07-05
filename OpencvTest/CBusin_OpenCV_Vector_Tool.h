/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: CBusin_OpenCV_Vector_Tool.h
* @brief: 向量运算，float点可代表一个向量
* @author:	minglu2
* @version: 1.0
* @date: 2018/08/27
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2018/08/27	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once


#ifdef __cplusplus  
extern "C" {  
//包含C语言接口、定义或头文件
#endif  
#ifdef __cplusplus  
}  
#endif  
//引用C++头文件：先是标准库头文件，后是项目头文件
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
//宏定义

//类型定义
class  CBusin_OpenCV_Vector_Tool
{
public:
	typedef cv::Point2f Type_Vector;
	static CBusin_OpenCV_Vector_Tool& instance();
	//由两点获取对应的向量,向量AB
	Type_Vector get_vector_from_points(const Point& point_A, const Point& point_B);
	//对于直角三角形ABC，根据点A和点C以及点B.x求B.y
	int get_PointB_of_triangle_ABC(const Point& point_A, const Point& point_C, Point& point_B);
	/************************************
	* Method:    is_special_right_triangle
	* Brief:  判定其是否能够组成边长比为3:4:5的直角三角形，并且角A为36.87度，角B为53.13度，角C为直角
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
//函数原型定义
