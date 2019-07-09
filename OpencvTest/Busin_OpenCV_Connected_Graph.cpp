/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Connected_Graph.cpp
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
#include "Busin_OpenCV_Connected_Graph.h"
#include <iostream>
#include <stack>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "File_System_Tool.h"
#define __CLASS_FUNCTION__ ((std::string("CBusin_OpenCV_Connected_Graph::") + std::string(__FUNCTION__)).c_str()) 

CBusin_OpenCV_Connected_Graph::~CBusin_OpenCV_Connected_Graph()
{

}

void CBusin_OpenCV_Connected_Graph::get_writing_connected_contours(const Mat& mat_src_binary_gray, vector<vector<Point> >& vec_contours)
{
	size_t nNum_black_points = 0;
	size_t nNum_connected_contours = 0; //��ͨ������Ŀ��ԽСԽ��
	Mat mat_tmp_binary_gray = mat_src_binary_gray.clone();
	for (int nRow = 0; nRow != mat_tmp_binary_gray.rows; ++nRow)
	{
		for (int nCol = 0; nCol != mat_tmp_binary_gray.cols; ++nCol)
		{
			if (mat_tmp_binary_gray.at<uchar>(nRow, nCol) < 10)
			{//��ǰ��Ϊ��ɫ
				vector<Point> vec_points; //��ǰĿ�����ص��Ӧ�Ĵ���·����ͨͼ
//				int ret = recursive_search_connected_graph(mat_tmp_binary_gray, nRow, nCol, vec_points);
				int ret =  non_recursive_traversal(mat_tmp_binary_gray, nRow, nCol, vec_points);
				if (0 == ret)
				{
					vec_contours.push_back(vec_points);
					nNum_black_points += vec_points.size();
					++nNum_connected_contours;
				}
				else
				{
					printf("%s | fail to get connected graph, ret:%d, row:%d, col:%d\n", __CLASS_FUNCTION__, ret, nRow, nCol);
				}
			}
		}
	}
#if _DEBUG
	printf("%s | Number of black points:%d, Number of contours:%d\n", __CLASS_FUNCTION__, nNum_black_points, nNum_connected_contours);
#endif
}

CBusin_OpenCV_Connected_Graph& CBusin_OpenCV_Connected_Graph::instance()
{
	static CBusin_OpenCV_Connected_Graph obj;
	return obj;
}

int CBusin_OpenCV_Connected_Graph::recursive_search_connected_graph(Mat& mat_src_binary_gray, int nRow, int nCol, vector<Point>& vec_points)
{
	//�ж�i�ĺϷ���
	if (nRow < 0 || nRow >= mat_src_binary_gray.rows)
	{
		return 10106;
	}
	if (nCol < 0 || nCol >= mat_src_binary_gray.cols)
	{
		return 10106;
	}
	//�ж���ǰ���Ƿ�ΪĿ���
	if (mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
	{
		//����������
		vec_points.push_back(Point(nCol, nRow));
		//���������Ϊ��ɫ
		mat_src_binary_gray.at<uchar>(nRow, nCol) = 255;
	}
	else
	{
		return 10106;
	}
	//�������ұߵ�
	int ret = recursive_search_connected_graph(mat_src_binary_gray, nRow, nCol + 1, vec_points);
// 	if (0 == ret)
// 	{//��һ����ΪĿ���
// 		vec_points.push_back(Point(nCol, nRow));
// 	}
	//�������±ߵ�
	ret = recursive_search_connected_graph(mat_src_binary_gray, nRow + 1, nCol, vec_points);
// 	if (0 == ret)
// 	{//��һ����ΪĿ���
// 		vec_points.push_back(Point(nCol, nRow));
// 	}
	//��������ߵ�
	ret = recursive_search_connected_graph(mat_src_binary_gray, nRow, nCol - 1, vec_points);
// 	if (0 == ret)
// 	{//��һ����ΪĿ���
// 		vec_points.push_back(Point(nCol, nRow));
// 	}
	ret = recursive_search_connected_graph(mat_src_binary_gray, nRow - 1, nCol, vec_points);
	//�������ϱߵ�
// 	if (0 == ret)
// 	{//��һ����ΪĿ���
// 		vec_points.push_back(Point(nCol, nRow));
// 	}
	return 0;
}

void CBusin_OpenCV_Connected_Graph::test_connected_graph_arr()
{
	//0��ʾĿ��
	uchar arr_data_2[6][6] = {
		255, 0, 255, 255, 0, 255,
		0,   0,  0,   0,  0, 255,
		0,  255, 0,  255, 0, 255,
		0,  255, 0,  255, 0, 255,
		0,  255, 0,  255, 0, 255,
		0,  255,255, 255, 0, 255
	};
	Mat mat_src_binary_gray(6, 6, CV_8UC1, arr_data_2);
	std::cout << __CLASS_FUNCTION__ << " | mat_src_binary_gray:" << mat_src_binary_gray << endl;
	std::cout << __CLASS_FUNCTION__ << " | mat_src_binary_gray[0][3]:" << (int)mat_src_binary_gray.at<uchar>(0, 3) << endl;
	vector<vector<Point> > vec_contrours;
	get_writing_connected_contours(mat_src_binary_gray, vec_contrours);
	for (int i =0; i != vec_contrours.size(); ++i)
	{
		std::cout << "contour[" << i << "]:"; 
		for (int j = 0; j != vec_contrours[i].size(); ++j)
		{
			std::cout << Point(vec_contrours[i][j].y, vec_contrours[i][j].x) << " ";
		}
		std::cout << endl;
	}
}

void CBusin_OpenCV_Connected_Graph::test_connected_graph_img()
{
	const string str_dst_dir = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_result/connected_graph_test/";
	const string str_src_dir = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_src/connected_graph_test/";
	//��ȡĿ¼��ԭ�ļ�
	vector<string> vec_src_file_path;
	int ret = sp_boost::get_files_path_list(str_src_dir, vec_src_file_path);
	if (ret)
	{
		std::cout << __CLASS_FUNCTION__ << " | fail to get files" << std::endl;
		return;
	}
	for (int idx = 0; idx != vec_src_file_path.size(); ++idx)
	{
		const string& str_src_img_path = vec_src_file_path[idx];
		Mat mat_src_bgr = imread(str_src_img_path, 1);
		Mat mat_src_gray;
		cv::cvtColor(mat_src_bgr, mat_src_gray, CV_BGR2GRAY);
		//��ֵ��
		Mat mat_src_binary_gray;
		cv::threshold(mat_src_gray, mat_src_binary_gray, 252, 255, THRESH_BINARY);
		vector<vector<Point> > vec_contrours;
		get_writing_connected_contours(mat_src_binary_gray, vec_contrours);
		size_t nNum_target_point = 40;//Ŀ�����Ŀ��ֵ
		//��ԭͼ�й�����㣺���˵�Ԫ�غ��ٵ���ͨͼ
		for (int i = 0; i != vec_contrours.size(); ++i)
		{
			//����ͨͼ��Ԫ����ĿС��Ŀ�����ֵʱ������Ϊ����ͨͼ�е�Ԫ�ض�Ϊ���
			if (vec_contrours[i].size() < nNum_target_point)
			{//Ϊ��㣬��ԭͼ�ж�Ӧλ������Ϊ��ɫ
				for (int j = 0; j != vec_contrours[i].size(); ++j)
				{
					mat_src_binary_gray.at<uchar>(vec_contrours[i][j]) = 255;
				}
			}
		}
		const string str_dst_path = str_dst_dir + "TargetPoints_"+ boost::lexical_cast<string>(nNum_target_point)
			+ "_" + boost::filesystem::path(str_src_img_path).filename().string();
		imwrite(str_dst_path, mat_src_binary_gray);
	}
	
}

int CBusin_OpenCV_Connected_Graph::non_recursive_traversal(Mat& mat_src_binary_gray, int nRow, int nCol, vector<Point>& vec_contour)
{

	std::stack<cv::Point> stk_path; //������¼����ʱ��·��
	cv::Point root;
	//����ǰ�����ջ��
	root = Point(nCol, nRow);
	stk_path.push(root);
	while (!stk_path.empty())
	{
		cv::Point point_current = stk_path.top();
		stk_path.pop();
		//�ж���ǰ���Ƿ�ΪĿ�����ص㣬��������������һ��
		if (false == is_target_pixel(mat_src_binary_gray, point_current.y, point_current.x))
		{
			continue;
		}
		//����������
		vec_contour.push_back(point_current);
		//�޸�Ϊ��Ŀ���
		mat_src_binary_gray.at<uchar>(point_current) = 255;
		//���ʳɹ�

		//Ϊ��ģ��ݹ���ҡ��¡����ϣ����ڷ���ջʱ��˳��Ϊ���ϡ����¡���
		if (is_target_pixel(mat_src_binary_gray, point_current.y - 1, point_current.x))
		{//��
			stk_path.push(Point(point_current.x, point_current.y - 1));
		}
		if (is_target_pixel(mat_src_binary_gray, point_current.y, point_current.x - 1))
		{//��
			stk_path.push(Point(point_current.x - 1, point_current.y));
		}
		if (is_target_pixel(mat_src_binary_gray, point_current.y + 1, point_current.x))
		{//��
			stk_path.push(Point(point_current.x, point_current.y + 1));
		}
		if (is_target_pixel(mat_src_binary_gray, point_current.y, point_current.x + 1))
		{//��
			stk_path.push(Point(point_current.x + 1, point_current.y));
		}
	
	}
	return 0;
}

bool CBusin_OpenCV_Connected_Graph::is_target_pixel(const Mat& mat_src_binary_gray, int nRow, int nCol)
{
	//�ж�i�ĺϷ���
	if (nRow < 0 || nRow >= mat_src_binary_gray.rows)
	{
		return false;
	}
	if (nCol < 0 || nCol >= mat_src_binary_gray.cols)
	{
		return false;
	}
	//�ж���ǰ���Ƿ�ΪĿ���
	if (mat_src_binary_gray.at<uchar>(nRow, nCol) > 10)
	{
		return false;
	}
	return true;
}
