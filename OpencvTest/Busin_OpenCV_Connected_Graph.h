/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Connected_Graph.h
* @brief: 遍历图片中黑色点构成的连通区域
* @author:	minglu2
* @version: 1.0
* @date: 2019/07/05
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/07/05	<td>minglu	<td>Create head file </tr>
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
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//宏定义

//类型定义
class CBusin_OpenCV_Connected_Graph
{
public:
	~CBusin_OpenCV_Connected_Graph();
	void get_writing_connected_contours(const Mat& mat_src_binary_gray, vector<vector<Point> >& vec_contours);
	static CBusin_OpenCV_Connected_Graph& instance();
	int recursive_search_connected_graph(Mat& mat_src_binary_gray, int nRow, int nCol, vector<Point>& vec_points);
	void test_connected_graph_arr();
	void test_connected_graph_img();
	int non_recursive_traversal(Mat& mat_src_binary_gray, int nRow, int nCol, vector<Point>& vec_contour);
	bool is_target_pixel(const Mat& mat_src_binary_gray, int nRow, int nCol);
protected:
private:
};
//函数原型定义
