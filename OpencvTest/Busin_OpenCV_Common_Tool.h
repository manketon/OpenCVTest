/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Common_Tool.h
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2018/09/10
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2018/09/10	<td>minglu	<td>Create head file </tr>
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
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <cmath>
#include <boost/lexical_cast.hpp>
#include <opencv2/core/core_c.h>
#include "CBusin_Opencv_Transform_Tool.h"
#include "File_System_Tool.h"
//宏定义
using namespace cv;
using namespace std;
//类型定义
class CBusin_OpenCV_Common_Tool
{
public:
	static CBusin_OpenCV_Common_Tool& instance();
	int get_my_RotatedRect(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3, RotatedRect& rRect);
	int draw_lines(const Mat& mat_src, const Point2f* pArr, size_t n, const Scalar& color, const string& str_win_name);
	int test_get_binary_gray_mat(const string& str_images_dir);
	int test_dilate(const string& str_img_path);
	int test_shrink(const string& str_img_path);
	//缩放图像
	int test_resize(const string& str_img_path);
	void scaleIntervalSampling(const Mat &src, Mat &dst, double xRatio, double yRatio);
	//基于局部均值的图像缩小
	void scalePartAverage(const Mat &src, Mat &dst, double xRatio, double yRatio);
	void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p);

	void shrink_by_part_average(const Mat &src, Mat &dst, double xRatio, double yRatio);

	void average_for_3_channel(const Mat &img, const Point& a, const Point& b, Vec3b &p);
	void average_for_1_channel(const Mat &img, const Point& a, const Point& b, uchar &nAverage);
	/************************************
	* Method:    detect_circles
	* Brief:  函数说明
	* Access:    public 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & src_bgr_mat -[in]  BGR图像
	*Parameter: double dMin_centers_dist -[in] 为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心 
	*Parameter: vector<Vec3f> & circles -[out]  为输出圆向量，每个向量包括三个浮点型的元素——圆心横坐标，圆心纵坐标和圆半径
	*Parameter: double dMax_canny_threshold -[in] 为边缘检测时使用Canny算子的高阈值  
	*Parameter: double dCircle_center_threshold -[int] 圆心检测阈值 
	*Parameter: int minRadius -[in] 能检测到的最小圆半径, 默认为0 
	*Parameter: int maxRadius -[in/out]  能检测到的最大圆半径, 默认为0
	************************************/
	int detect_circles(const Mat& src_bgr_mat, const Rect& roi, double dMin_centers_dist, vector<Vec3f>& circles, double dMax_canny_threshold = 100, double dCircle_center_threshold = 100,int minRadius = 0, int maxRadius = 0);
	int test_detect_circles(const std::string& str_img_path);
	int test_transform();
	double get_euclidean(const cv::Point& pointO, const cv::Point& pointA);
	int deal_nc_point(const Point& point_00_before, const Point& point_00_after
		, const Point2f& point_src_from_nc, const cv::Mat& warp_mat, const float& fScale_pic_devide_mm_before
		, const float& fScale_pic_devide_mm_after, Point2f& point_dst_to_nc, std::string& str_kernel_err_reason);

	cv::Point2f get_dst_point_after_affine(const cv::Point2f& src_point, const Mat& affine_transform_mat);

	bool is_special_right_triangle(const cv::Point& point_A, const cv::Point& point_B
		, const cv::Point& point_C, const double& dBias_degree);
	/************************************
	* Method:    change_contrast_and_brightness
	* Brief:  通过改变原图的颜色对比度和亮度来得到新的图片
	原理：mat_dst_image(i,j) = udContrast*mat_src_bgr(i,j) + nBrightness
	* Access:    public 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_src_bgr -[in]  
	*Parameter: double udContrast -[in]  对比度（即增益），要求大于0;属于[1.0:3.0]时，效果好点。
	*Parameter: int nBrightness -[in] 亮度（即偏置）,正负数都可以  
	*Parameter: Mat & mat_dst_image -[in/out]  
	************************************/
	int change_contrast_and_brightness(const Mat& mat_src_bgr, double udContrast, int nBrightness, Mat& mat_dst_image);
	static void callback_change_contrast_and_brightness(int pos, void* userdata);
	int test_change_contrast_and_brightness(const string& str_img_path);
	int test_wenzi_G_code(const string& str_img_path);
	int test_shrink_mat();
	//利用Catmull-Rom算法拟合曲
	int test_Catmull_Rom();
	//使用CatmullRom插样算法来获取点p1和p2直接曲线上的点
	int get_CatmullRom_points(const cv::Point2f& p0, const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3
		, int nGranularity, std::vector<cv::Point2f>& vec_curve_points);

protected:
	/************************************
	* Method:    get_binary_gray_mat
	* Brief:  获取原始矩阵对应的二值灰度矩阵（先灰度化再二值化）
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const cv::Mat & mat_src -[in] 原始BGR图片 
	*Parameter: int nThreshold_binary -[in]  二值化阈值（大于此阈值则变为255，否则变为0）
	*Parameter: Mat & mat_dst_binary -[out] 二值灰度矩阵 （白色：255，黑色：0） 
	************************************/
	int get_binary_gray_mat(const cv::Mat& mat_src, int nThreshold_binary, Mat& mat_dst_binary);
	int create_nc_code(const Mat& mat_dst_binary, const Point& point_00 
		, float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data);
	int get_nc_effective_data(const Mat& mat_binary, const Point& point_00, size_t& nLastN , float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data);
	/************************************
	* Method:    get_nc_head
	* Brief:  获取文件头及下一行位置
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: size_t & nLastN -[in/out]  
	************************************/
	string get_nc_head(size_t& nLastN);
	/************************************
	* Method:    get_nc_tail
	* Brief:  根据当前行下标来构造NC代码的尾部
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: const size_t & nLastN -[in] 当前总行数+1  
	************************************/
	string get_nc_tail(const size_t& nLastN);
	int rotate_and_shrink_binary_writting_mat(const Mat& mat_src_binary_gray, const float& fMax_width_pxl
		, const float& fMax_height_pxl, const float& fScale_pic_devide_mm, const Point& point_00, Mat& mat_binary_gray_shrinked);
	int shrink_binary_mat_using_contours(const Mat& mat_src_binary, const float& fMax_width_pxl
		, const float& fMax_height_pxl, const Point& point_00, Mat& mat_binary_shrinked);
	Mat override_sub_mat(const Mat& m1, const Mat& m2, const Point& rio_loction_point);
	int  test_override_sub_mat();
	int remove_error_black_points(Mat& mat_binary);
	/************************************
	* Method:    calc_black_points_num
	* Brief:  计算在mat中，以此点为中心，x、y方向的步伐nStep的区域范围内，黑点的个数
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_binary -[in/out]  
	*Parameter: const Point & point_current -[in/out]  
	*Parameter: size_t nStep -[in/out]  
	************************************/
	int calc_black_points_num(const Mat& mat_binary, const Point& point_current, size_t nStep);
	int draw_lines(const Mat& mat_src, Point2f* pArr, size_t n, const string& str_img_path);

	//将矩形旋转至水平方向
	int rotate_writting_to_horizontal_direction(const Mat& src_mat, const RotatedRect& rRect
		, const float& fScale, const float& fScale_pic_devide_mm, Mat& mat_dst);

	/************************************
	* Method:    get_points_after_affined
	* Brief:  根据变换前的矩形顶点和仿射矩阵来求取变换后对应的矩形顶点
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_rotation -[in] 仿射矩阵 
	*Parameter: Point2f * pSrc_points -[in] 源点数组  
	*Parameter: size_t nPoints_num -[in] 源点数目
	*Parameter: Point2f * pDst_points -[out] 目标点数组 
	************************************/
	int get_points_after_affined(const Mat& mat_rotation, const Point2f* pSrc_points, size_t nPoints_num, Point2f* pDst_points);
	/************************************
	* Method:    get_rect_points
	* Brief:  根据RotatedRect提取外接矩形的四个顶点
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_src -[in] 源矩阵 
	*Parameter: const RotatedRect & rRect -[in] 最小外接矩形 
	*Parameter: Point2f * pts -[out] 最小外接矩形对应的 
	*Parameter: size_t nPoints_num -[in] 矩形顶点数目 
	************************************/
	int get_rect_points(const Mat& mat_src, const RotatedRect& rRect, Point2f* pts, size_t nPoints_num = 4);

	/************************************
	* Method:    sort_rect_points
	* Brief:  对矩形的顶点列表进行排序，以使得其点顺序为bottomLeft, topLeft, topRight, bottomRight
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  作为输入时：矩形的顶点数组；作为输出时：四个顶点被排序了
	*Parameter: size_t nPoints_num -[in] 顶点个数
	************************************/
	int sort_rect_points(Point2f* pRect_points, size_t nPoints_num);
	/************************************
	* Method:    increase_rect_size
	* Brief:  将矩形的四个顶点（顺序为bottomLeft, topLeft, topRight, bottomRight）分别在x和y方向上平移nExt_size_pxl个像素，以使得整个矩形范围变大
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  矩形顶点
	*Parameter: size_t nPoints_num -[in/out]  
	*Parameter: int nExt_size_pxl -[in/out]  
	************************************/
	int increase_rect_size(Point2f* pRect_points, size_t nPoints_num, int nExt_size_pxl);
	void get_connected_contours(const Mat& mat_src_binary_gray, vector<vector<Point> >& vec_contours);
	/************************************
	* Method:    traverse_connected_graph
	* Brief:  遍历连通图，将访问过的目标像素设置为255，即将黑色像素设置为白色
	* Access:    protected 
	* Returns:   void
	* Qualifier:
	*Parameter: Mat & mat_src_binary_gray -[in/out]  
	*Parameter: const Point & point_start -[in/out]  
	*Parameter: vector<Point> & vec_points -[in/out]  
	************************************/
	void traverse_connected_graph(Mat& mat_src_binary_gray, const Point& point_start, vector<Point>& vec_points);

	int get_nc_effective_data(const vector<vector<Point> >& vec_contours, const Point& point_00, size_t& nLastN
		, float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data);
	int draw_curve_with_CatmullRom()
	{
		return 0;
	}
	CBusin_OpenCV_Common_Tool();
	CBusin_OpenCV_Common_Tool(const CBusin_OpenCV_Common_Tool&);
	CBusin_OpenCV_Common_Tool& operator=(const CBusin_OpenCV_Common_Tool&);
private:
};
//函数原型定义
