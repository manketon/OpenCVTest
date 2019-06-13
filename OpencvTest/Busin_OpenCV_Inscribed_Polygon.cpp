/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Inscribed_Polygon.cpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2019/06/13
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/06/13	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "Busin_OpenCV_Inscribed_Polygon.h"
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include "Busin_OpenCV_Common_Tool.h"


void CInscribed_Polygon_Tool::test_max_inscribed_rect_using_traversing_for_rotated2()
{
	double Time = (double)cvGetTickCount();
	const string str_img_path = "./images_for_MER/cloud.png";
	Mat mat_src_bgr = imread(str_img_path, IMREAD_COLOR);
	//将原图转换为灰度图
	Mat mat_src_gray;
	cvtColor(mat_src_bgr, mat_src_gray, COLOR_BGR2GRAY);
	//查找最大面积的轮廓
	//求二值灰度矩阵(黑色为底)
	Mat mat_src_binary_gray;
	threshold(mat_src_gray, mat_src_binary_gray, 100, 255,THRESH_BINARY);
	cv::imshow("mat_src_binary_gray", mat_src_binary_gray);
	//最大面积的内接矩形肯定在最大面积的轮廓中，故先查找最大面积的轮廓
	//查找最大面积的轮廓
	vector<Point> vec_max_area_contour;
	int ret = FindBiggestContour(mat_src_binary_gray, vec_max_area_contour);
	if (ret)
	{
		printf("%s | error\n", __FUNCTION__);
		return;
	}
	//查找最大面积轮廓对应的最小外接矩形，并以此最小外接矩形的中心作为旋转中心
	RotatedRect rRect_min_area = cv::minAreaRect(vec_max_area_contour);
	Mat mat_with_max_area;//最大面积时对应的旋转图片
	Rect rect_upRight_MER_with_max_area; //最大面积时对应的旋转图片中的最大内接upRight矩形
	double dDegree_with_max_area = 0;
	//子矩形：人工确定最大内接矩形肯定包含的子矩形，便于降低时间复杂度
	Rect rect_sub(163, 177, 117, 82);
	Point2f points_arr_src[4] = {
		Point(rect_sub.x, rect_sub.y + rect_sub.height)
		, Point(rect_sub.x, rect_sub.y)
		, Point(rect_sub.x + rect_sub.width, rect_sub.y)
		, Point(rect_sub.x + rect_sub.width, rect_sub.y + rect_sub.height)};

	CBusin_OpenCV_Common_Tool::instance().draw_lines(mat_src_binary_gray, points_arr_src, 4, Scalar(0), "src img with sub rect");
	vector<Mat> vec_mat_rotated(91);
	vector<Rect> vec_rect_upRight_MER(91);
	vector<boost::thread*> vec_thread_calc_upRight_MER;
	//对原图进行无损失旋转，再对旋转所得图片进行穷举最大内接upRight矩形，记录面积最大时的情况
	int nStep = 1;
	for (double dDegree = 0; dDegree <= 90; dDegree += nStep)
	{
		boost::thread* pThread = new boost::thread(boost::bind(&CInscribed_Polygon_Tool::func_thread_upRight_MER, this
			, mat_src_bgr, rRect_min_area.center, dDegree, rect_sub
			, boost::ref(vec_mat_rotated[(int)dDegree]), boost::ref(vec_rect_upRight_MER[(int)dDegree])));
		vec_thread_calc_upRight_MER.push_back(pThread);
	}

	//等待所有线程结束
	for (int idx = 0; idx != vec_thread_calc_upRight_MER.size(); ++idx)
	{
		if (vec_thread_calc_upRight_MER[idx]->joinable())
		{
			vec_thread_calc_upRight_MER[idx]->join();
		}
		else
		{
			std::cout << __FUNCTION__ << " | thread is not joinable, id:" << vec_thread_calc_upRight_MER[idx]->get_id() << endl;
		}
	}
	for (int idx = 0; idx != vec_rect_upRight_MER.size(); ++idx)
	{
		//将每个线程的结果放入列表中
		if (vec_rect_upRight_MER[idx].area() > rect_upRight_MER_with_max_area.area())
		{
			mat_with_max_area = vec_mat_rotated[idx].clone();
			rect_upRight_MER_with_max_area = vec_rect_upRight_MER[idx];
			dDegree_with_max_area = idx * nStep; //不严谨
		}
	}
	Time = (double)cvGetTickCount() - Time;
	printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//毫秒

	//最大面积对应的矩阵中画出矩形
	cv::rectangle(mat_with_max_area, rect_upRight_MER_with_max_area, Scalar(0, 0, 0), 1, LINE_8,0);
	std::cout << __FUNCTION__ << " | MER:" << rect_upRight_MER_with_max_area << ", area:" << rect_upRight_MER_with_max_area.area() 
		<< ", dDegree_with_max_area:" << dDegree_with_max_area << endl;
	//此时的最大面积矩阵逆向旋转（注意：此时图片的分辨率发生了变化，并且应该以旋转后的中心点来旋转）
	Mat mat_withMER = CBusin_Opencv_Transform_Tool::instance().rotate_image_without_loss(
		mat_with_max_area, rRect_min_area.center/*中心点旋转错误*/, -1 * dDegree_with_max_area, 1, Scalar(0, 0, 0));
	cv::imshow("mat src with MER", mat_withMER);
	cv::imwrite(str_img_path + "_withMER.jpg", mat_withMER);
	// 		cv::imshow("mat_max_area with MER", mat_with_max_area);
	// 		cv::imwrite(str_img_path + "_withMER.jpg", mat_with_max_area);
	cv::waitKey(0);
}

int CInscribed_Polygon_Tool::get_upRight_MER_using_traversing3(const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_upRight_MER)
{
	if (mat_src_bgr.type() != CV_8UC3)
	{
		std::cout << __FUNCTION__ << " | type of mat_src_bgr:" << mat_src_bgr.type() << ", is not CV_8UC3:" << CV_8UC3 << std::endl; 
		return -1;
	}
	//将BRG图片转换为灰度图
	Mat mat_src_gray;
	cv::cvtColor(mat_src_bgr, mat_src_gray, COLOR_BGR2GRAY);
	//求二值灰度矩阵(黑色为底)
	Mat mat_src_binary_gray;
	threshold(mat_src_gray, mat_src_binary_gray, 100, 255,THRESH_BINARY);
	//最大面积的内接矩形肯定在最大面积的轮廓中，故先查找最大面积的轮廓
	//查找最大面积的轮廓
	vector<Point> vec_max_area_contour;
	int ret = FindBiggestContour(mat_src_binary_gray, vec_max_area_contour);
	if (ret)
	{
		printf("%s | error\n", __FUNCTION__);
		return -1;
	}

	//计算最大面积的轮廓对应的最小外包矩形
	Rect rect_bbox =  cv::boundingRect(vec_max_area_contour);
	if (rect_bbox.empty() || rect_bbox.height <= 0 || rect_bbox.width <= 0)
	{
		std::cout << __FUNCTION__ << " | failed to get bbox, rect_bbox:" << rect_bbox << endl; 
		return -1;
	}
	double dMax_area = INT_MIN;
	int flag_x_min = 0, flag_x_max = 0, flag_y_min = 0, flag_y_max = 0;
	//以最小外接矩形的边界扩展后的值作为x、y的取值范围，注意：最小外接矩形的边界可能和轮廓边缘重合，故要扩展
	int nXmin = rect_bbox.x - 1, nXmax = rect_bbox.x + rect_bbox.width - 1 + 1; 
	int nYmin = rect_bbox.y - 1, nYmax = rect_bbox.y + rect_bbox.height - 1 + 1;
	int nSub_rect_MinX = rect_sub.x, nSub_rect_MaxX = rect_sub.x + rect_sub.width - 1; 
	int nSub_rect_MinY = rect_sub.y, nSub_rect_MaxY = rect_sub.y + rect_sub.height - 1;
	BG_Element** x_white_line_ptr_Arr = NULL; //x_white_line_ptr_Arr[i]后面跟着的表节点表示在轮廓内，x=i中，线段[(i,begin),（i,end)]之间都为白色
	BG_Element** y_white_line_ptr_Arr = NULL;

	//遍历二值灰度矩阵，分别获取X/Y方向上的白色线段区间
	ret = get_white_line_arr(mat_src_binary_gray, x_white_line_ptr_Arr, y_white_line_ptr_Arr, nXmin, nXmax, nYmin, nYmax);
	int nMin_dist_X = 2; //X方向两边界的最小间隔
	int nMin_dist_Y = 2; //Y方向上两边界的最小间隔
	for (int i = nXmin; i <= nXmax && i <= nSub_rect_MinX; ++i)
	{ 
		for (int j = /*i + nMin_dist_X*/ nSub_rect_MaxX; j <= nXmax; ++j)
		{
			for (int m = nYmin; m <= nYmax  && m <= nSub_rect_MinY; ++m)
			{
				//判定三条线所得的两个顶点是否在轮廓外
				//根据灰度值来判定点是否在轮廓外
				if (mat_src_binary_gray.at<uchar>(m, i) == 0)
				{//在轮廓外
					continue;
				}
				if (mat_src_binary_gray.at<uchar>(m, j) == 0)
				{
					continue;
				}
				for (int n = /*m + nMin_dist_Y*/  nSub_rect_MaxY; n <= nYmax; ++n)
				{
					if (mat_src_binary_gray.at<uchar>(n, i) == 0)
					{//在轮廓外
						continue;
					}
					if (mat_src_binary_gray.at<uchar>(n, j) == 0)
					{
						continue;
					}
					//使用rect_edge_has_black成功
					if (rect_edge_has_black2(mat_src_binary_gray, x_white_line_ptr_Arr, y_white_line_ptr_Arr, i, j, m, n) == false)
					{//四条边上都无黑点，则说明此矩形有效
						//计算此四边所组成的矩形的面积
						double dArea_tmp = (j - i + 1)*(n - m + 1);
						if (dMax_area < dArea_tmp)
						{
							dMax_area = dArea_tmp;
							flag_x_min = i;
							flag_x_max = j;
							flag_y_min = m;
							flag_y_max = n;
						}
					}
					else
					{//有黑点在四边所组成的矩形上,则不再当前方向上继续扩展边界
						break;
					}
				}
			}
		}
	}
	ret = release_white_line_arr(mat_src_binary_gray.size(), x_white_line_ptr_Arr, y_white_line_ptr_Arr);
	std::cout << __FUNCTION__ << " | flag_x_min:" << flag_x_min << ", flag_x_max:" << flag_x_max 
		<< ", flag_y_min:" << flag_y_min << ",flag_y_max:" << flag_y_max << ", dMax_area:" << dMax_area << endl;
	rect_upRight_MER =  Rect(flag_x_min, flag_y_min, flag_x_max - flag_x_min + 1, flag_y_max - flag_y_min + 1);
	return 0;
}
