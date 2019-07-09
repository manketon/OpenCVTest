/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Connected_Graph.h
* @brief: ����ͼƬ�к�ɫ�㹹�ɵ���ͨ����
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
#pragma once
#ifdef __cplusplus  
extern "C" {  
//����C���Խӿڡ������ͷ�ļ�
#endif  
#ifdef __cplusplus  
}  
#endif  
//����C++ͷ�ļ������Ǳ�׼��ͷ�ļ���������Ŀͷ�ļ�
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//�궨��

//���Ͷ���
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
//����ԭ�Ͷ���
