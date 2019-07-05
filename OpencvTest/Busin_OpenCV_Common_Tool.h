/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Common_Tool.h
* @brief: ���˵���ļ����ܡ���; (Comment)��
* @author:	minglu2
* @version: 1.0
* @date: 2018/09/10
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
*  <tr> <td>1.0	    <td>2018/09/10	<td>minglu	<td>Create head file </tr>
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
//�궨��
using namespace cv;
using namespace std;
//���Ͷ���
class CBusin_OpenCV_Common_Tool
{
public:
	static CBusin_OpenCV_Common_Tool& instance();
	int get_my_RotatedRect(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3, RotatedRect& rRect);
	int draw_lines(const Mat& mat_src, const Point2f* pArr, size_t n, const Scalar& color, const string& str_win_name);
	int test_get_binary_gray_mat(const string& str_images_dir);
	int test_dilate(const string& str_img_path);
	int test_shrink(const string& str_img_path);
	//����ͼ��
	int test_resize(const string& str_img_path);
	void scaleIntervalSampling(const Mat &src, Mat &dst, double xRatio, double yRatio);
	//���ھֲ���ֵ��ͼ����С
	void scalePartAverage(const Mat &src, Mat &dst, double xRatio, double yRatio);
	void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p);

	void shrink_by_part_average(const Mat &src, Mat &dst, double xRatio, double yRatio);

	void average_for_3_channel(const Mat &img, const Point& a, const Point& b, Vec3b &p);
	void average_for_1_channel(const Mat &img, const Point& a, const Point& b, uchar &nAverage);
	/************************************
	* Method:    detect_circles
	* Brief:  ����˵��
	* Access:    public 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & src_bgr_mat -[in]  BGRͼ��
	*Parameter: double dMin_centers_dist -[in] ΪԲ��֮�����С���룬�����⵽������Բ��֮�����С�ڸ�ֵ������Ϊ������ͬһ��Բ�� 
	*Parameter: vector<Vec3f> & circles -[out]  Ϊ���Բ������ÿ�������������������͵�Ԫ�ء���Բ�ĺ����꣬Բ���������Բ�뾶
	*Parameter: double dMax_canny_threshold -[in] Ϊ��Ե���ʱʹ��Canny���ӵĸ���ֵ  
	*Parameter: double dCircle_center_threshold -[int] Բ�ļ����ֵ 
	*Parameter: int minRadius -[in] �ܼ�⵽����СԲ�뾶, Ĭ��Ϊ0 
	*Parameter: int maxRadius -[in/out]  �ܼ�⵽�����Բ�뾶, Ĭ��Ϊ0
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
	* Brief:  ͨ���ı�ԭͼ����ɫ�ԱȶȺ��������õ��µ�ͼƬ
	ԭ��mat_dst_image(i,j) = udContrast*mat_src_bgr(i,j) + nBrightness
	* Access:    public 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_src_bgr -[in]  
	*Parameter: double udContrast -[in]  �Աȶȣ������棩��Ҫ�����0;����[1.0:3.0]ʱ��Ч���õ㡣
	*Parameter: int nBrightness -[in] ���ȣ���ƫ�ã�,������������  
	*Parameter: Mat & mat_dst_image -[in/out]  
	************************************/
	int change_contrast_and_brightness(const Mat& mat_src_bgr, double udContrast, int nBrightness, Mat& mat_dst_image);
	static void callback_change_contrast_and_brightness(int pos, void* userdata);
	int test_change_contrast_and_brightness(const string& str_img_path);
	int test_wenzi_G_code(const string& str_img_path);
	int test_shrink_mat();
protected:
	/************************************
	* Method:    get_binary_gray_mat
	* Brief:  ��ȡԭʼ�����Ӧ�Ķ�ֵ�ҶȾ����ȻҶȻ��ٶ�ֵ����
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const cv::Mat & mat_src -[in] ԭʼBGRͼƬ 
	*Parameter: int nThreshold_binary -[in]  ��ֵ����ֵ�����ڴ���ֵ���Ϊ255�������Ϊ0��
	*Parameter: Mat & mat_dst_binary -[out] ��ֵ�ҶȾ��� ����ɫ��255����ɫ��0�� 
	************************************/
	int get_binary_gray_mat(const cv::Mat& mat_src, int nThreshold_binary, Mat& mat_dst_binary);
	int create_nc_code(const Mat& mat_dst_binary, const Point& point_00 
		, float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data);
	int get_nc_effective_data(const Mat& mat_binary, const Point& point_00, size_t& nLastN , float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data);
	/************************************
	* Method:    get_nc_head
	* Brief:  ��ȡ�ļ�ͷ����һ��λ��
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: size_t & nLastN -[in/out]  
	************************************/
	string get_nc_head(size_t& nLastN);
	/************************************
	* Method:    get_nc_tail
	* Brief:  ���ݵ�ǰ���±�������NC�����β��
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: const size_t & nLastN -[in] ��ǰ������+1  
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
	* Brief:  ������mat�У��Դ˵�Ϊ���ģ�x��y����Ĳ���nStep������Χ�ڣ��ڵ�ĸ���
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_binary -[in/out]  
	*Parameter: const Point & point_current -[in/out]  
	*Parameter: size_t nStep -[in/out]  
	************************************/
	int calc_black_points_num(const Mat& mat_binary, const Point& point_current, size_t nStep);
	int draw_lines(const Mat& mat_src, Point2f* pArr, size_t n, const string& str_img_path);

	//��������ת��ˮƽ����
	int rotate_writting_to_horizontal_direction(const Mat& src_mat, const RotatedRect& rRect
		, const float& fScale, const float& fScale_pic_devide_mm, Mat& mat_dst);

	/************************************
	* Method:    get_points_after_affined
	* Brief:  ���ݱ任ǰ�ľ��ζ���ͷ����������ȡ�任���Ӧ�ľ��ζ���
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_rotation -[in] ������� 
	*Parameter: Point2f * pSrc_points -[in] Դ������  
	*Parameter: size_t nPoints_num -[in] Դ����Ŀ
	*Parameter: Point2f * pDst_points -[out] Ŀ������� 
	************************************/
	int get_points_after_affined(const Mat& mat_rotation, const Point2f* pSrc_points, size_t nPoints_num, Point2f* pDst_points);
	/************************************
	* Method:    get_rect_points
	* Brief:  ����RotatedRect��ȡ��Ӿ��ε��ĸ�����
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_src -[in] Դ���� 
	*Parameter: const RotatedRect & rRect -[in] ��С��Ӿ��� 
	*Parameter: Point2f * pts -[out] ��С��Ӿ��ζ�Ӧ�� 
	*Parameter: size_t nPoints_num -[in] ���ζ�����Ŀ 
	************************************/
	int get_rect_points(const Mat& mat_src, const RotatedRect& rRect, Point2f* pts, size_t nPoints_num = 4);

	/************************************
	* Method:    sort_rect_points
	* Brief:  �Ծ��εĶ����б����������ʹ�����˳��ΪbottomLeft, topLeft, topRight, bottomRight
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  ��Ϊ����ʱ�����εĶ������飻��Ϊ���ʱ���ĸ����㱻������
	*Parameter: size_t nPoints_num -[in] �������
	************************************/
	int sort_rect_points(Point2f* pRect_points, size_t nPoints_num);
	/************************************
	* Method:    increase_rect_size
	* Brief:  �����ε��ĸ����㣨˳��ΪbottomLeft, topLeft, topRight, bottomRight���ֱ���x��y������ƽ��nExt_size_pxl�����أ���ʹ���������η�Χ���
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  ���ζ���
	*Parameter: size_t nPoints_num -[in/out]  
	*Parameter: int nExt_size_pxl -[in/out]  
	************************************/
	int increase_rect_size(Point2f* pRect_points, size_t nPoints_num, int nExt_size_pxl);
	void get_connected_contours(const Mat& mat_src_binary_gray, vector<vector<Point> >& vec_contours);
	/************************************
	* Method:    traverse_connected_graph
	* Brief:  ������ͨͼ�������ʹ���Ŀ����������Ϊ255��������ɫ��������Ϊ��ɫ
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

	CBusin_OpenCV_Common_Tool();
	CBusin_OpenCV_Common_Tool(const CBusin_OpenCV_Common_Tool&);
	CBusin_OpenCV_Common_Tool& operator=(const CBusin_OpenCV_Common_Tool&);
private:
};
//����ԭ�Ͷ���
