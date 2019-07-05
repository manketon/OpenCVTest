/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Inscribed_Polygon.h
* @brief: ��ȡ�ڽӶ����
* @author:	minglu2
* @version: 1.0
* @date: 2019/05/23
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
*  <tr> <td>1.0	    <td>2019/05/23	<td>minglu	<td>Create head file </tr>
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
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "CBusin_Opencv_Transform_Tool.h"
using namespace std;
using namespace cv;
//�궨��
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
//���Ͷ���
class CInscribed_Polygon_Tool
{
public:
	static CInscribed_Polygon_Tool& instace();
	void test_max_inscribed_rect();
	//������������չ�㷨����ȡ����ڽӾ��Σ��˷����õ��Ľ������ȫ�����ţ�
	void test_max_inscribed_rect_only_Using_CEA();

	void test_max_inscribed_rect_using_traversing();
	
	void test_max_inscribed_rect_using_traversing_for_rotated();
	//���ںڵװ���ͼƬ��ͨ����ԭͼ��������ת�õ���ת���ͼ��Ȼ�����ת����ͼƬ������ٲ���������ڽӾ���
	//����������ڽӾ��μ�¼�������������ս������ԭͼ��Ӧ������ڽӾ���
	void test_max_inscribed_rect_using_traversing_for_rotated2();
	void func_thread_upRight_MER(const Mat& mat_src_bgr, const Point2f& center, double dDegree, const Rect& rect_sub_src
		, Mat& mat_rotated, Rect& rect_upRight_MER);
	void test_max_inscribed_rect_using_traversing2();
	void test_rect();
	int test3();

	//����ڽ�Բʵ��
	void test_max_inscribed_circle();
protected:
	int get_MER(const Mat& mat_src, string& str_err_reason);
	//ͼ��Ϊ�ڵ�
	int FindBiggestContour(const Mat& mat_src_gray, vector<Point>& vec_max_area_contour);

	/**
	* @brief ��ȡ��ͨ�����ڽӾ�
	* @param img:����ͼ�񣬵�ͨ����ֵͼ�����Ϊ8
	* @param center:��С��Ӿص�����
	* @return  ����ڽӾ���
	* ����������չ�㷨
	*/
	cv::Rect InSquare(Mat &img, const Point center);

	/**
	* @brief expandEdge ��չ�߽纯��
	* @param img:����ͼ�񣬵�ͨ����ֵͼ�����Ϊ8
	* @param edge  �߽����飬���4���߽�ֵ
	* @param edgeID ��ǰ�߽��
	* @return ����ֵ ȷ����ǰ�߽��Ƿ������չ
	*/
	bool expandEdge(const Mat & img, int edge[], const int edgeID);
	Rect get_upRight_MER_using_traversing(const Mat& mat_src_binary_gray,const Rect& rect_bbox);

	//���Ӿ��κ���С��Ӿ���֮������������ڽӾ���
	int get_upRight_MER_using_traversing2(const string&str_img_path, const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_MER);

	int get_upRight_MER_using_traversing3(const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_upRight_MER);


	//�ж������ı��Ƿ��к�ɫ�㣬��һ�ߺ��к�ɫ�㶼������
	bool rect_edge_has_black(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax);
	//�ж������ı��Ƿ��к�ɫ�㣬��һ�ߺ��к�ɫ�㶼������
	bool rect_edge_has_black2(const Mat& mat_src_binary_gray, BG_Element**&x_white_line_ptr_Arr, BG_Element**&y_while_line_ptr_Arr
		, int nXmin, int nXmax, int nYmin, int nYmax);

	//��x��yȷ�����ж���nBegin,nEnd��֮���Ƿ��кڵ�
	bool no_black_on_line(const BG_Element * p, size_t nBegin, size_t nEnd);
	//����������
	bool is_out_of_contour(const Mat& mat_src_binary_gray, const vector<Point>& contour, const Point& p0);
	int get_white_line_arr(const Mat& mat_src_binary_gray, BG_Element**& x_white_line_ptr_Arr, BG_Element**& y_white_line_ptr_Arr, int nXmin, int nXmax, int nYmin, int nYmax);
	int release_white_line_arr(const cv::Size& size_of_binary_gray_img, BG_Element**& x_white_line_ptr_Arr, BG_Element**& y_white_line_ptr_Arr);
private:
};
//����ԭ�Ͷ���
