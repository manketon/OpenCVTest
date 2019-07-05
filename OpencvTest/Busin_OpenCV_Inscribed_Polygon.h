/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Inscribed_Polygon.h
* @brief: 求取内接多边形
* @author:	minglu2
* @version: 1.0
* @date: 2019/05/23
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/05/23	<td>minglu	<td>Create head file </tr>
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
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "CBusin_Opencv_Transform_Tool.h"
using namespace std;
using namespace cv;
//宏定义
struct BG_Element
{
	BG_Element()
		: nBegin(0)
		, nEnd(0)
		, pNext(NULL)
	{

	}
	size_t nBegin;
	size_t nEnd;
	BG_Element * pNext;
};
//类型定义
class CInscribed_Polygon_Tool
{
public:
	static CInscribed_Polygon_Tool& instace();
	void test_max_inscribed_rect();
	//仅基于中心扩展算法来获取最大内接矩形（此方法得当的结果并非全局最优）
	void test_max_inscribed_rect_only_Using_CEA();

	void test_max_inscribed_rect_using_traversing();
	
	void test_max_inscribed_rect_using_traversing_for_rotated();
	//对于黑底白物图片，通过对原图进行逐渐旋转得到旋转后的图，然后对旋转所得图片进行穷举查找其最大内接矩形
	//将面积最大的内接矩形记录下来，这样最终结果就是原图对应的最大内接矩形
	void test_max_inscribed_rect_using_traversing_for_rotated2();
	void func_thread_upRight_MER(const Mat& mat_src_bgr, const Point2f& center, double dDegree, const Rect& rect_sub_src
		, Mat& mat_rotated, Rect& rect_upRight_MER);
	void test_max_inscribed_rect_using_traversing2();
	void test_rect();
	int test3();

	//最大内接圆实例
	void test_max_inscribed_circle();
protected:
	int get_MER(const Mat& mat_src, string& str_err_reason);
	//图像为黑底
	int FindBiggestContour(const Mat& mat_src_gray, vector<Point>& vec_max_area_contour);

	/**
	* @brief 求取连通区域内接矩
	* @param img:输入图像，单通道二值图，深度为8
	* @param center:最小外接矩的中心
	* @return  最大内接矩形
	* 基于中心扩展算法
	*/
	cv::Rect InSquare(Mat &img, const Point center);

	/**
	* @brief expandEdge 扩展边界函数
	* @param img:输入图像，单通道二值图，深度为8
	* @param edge  边界数组，存放4条边界值
	* @param edgeID 当前边界号
	* @return 布尔值 确定当前边界是否可以扩展
	*/
	bool expandEdge(const Mat & img, int edge[], const int edgeID);
	Rect get_upRight_MER_using_traversing(const Mat& mat_src_binary_gray,const Rect& rect_bbox);

	//在子矩形和最小外接矩形之间来查找最大内接矩形
	int get_upRight_MER_using_traversing2(const string&str_img_path, const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_MER);

	int get_upRight_MER_using_traversing3(const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_upRight_MER);


	//判定矩形四边是否含有黑色点，任一边含有黑色点都返回真
	bool rect_edge_has_black(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax);
	//判定矩形四边是否含有黑色点，任一边含有黑色点都返回真
	bool rect_edge_has_black2(const Mat& mat_src_binary_gray, BG_Element**&x_white_line_ptr_Arr, BG_Element**&y_while_line_ptr_Arr
		, int nXmin, int nXmax, int nYmin, int nYmax);

	//当x或y确定后，判定（nBegin,nEnd）之间是否含有黑点
	bool no_black_on_line(const BG_Element * p, size_t nBegin, size_t nEnd);
	//点在轮廓外
	bool is_out_of_contour(const Mat& mat_src_binary_gray, const vector<Point>& contour, const Point& p0);
	int get_white_line_arr(const Mat& mat_src_binary_gray, BG_Element**& x_white_line_ptr_Arr, BG_Element**& y_white_line_ptr_Arr, int nXmin, int nXmax, int nYmin, int nYmax);
	int release_white_line_arr(const cv::Size& size_of_binary_gray_img, BG_Element**& x_white_line_ptr_Arr, BG_Element**& y_white_line_ptr_Arr);
private:
};
//函数原型定义
