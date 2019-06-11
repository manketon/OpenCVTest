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
	static CBusin_OpenCV_Common_Tool& instance()
	{
		static CBusin_OpenCV_Common_Tool obj;
		return obj;
	}
	int get_my_RotatedRect(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3, RotatedRect& rRect);
	int draw_lines(const Mat& mat_src, const Point2f* pArr, size_t n, const Scalar& color)
	{
		Mat mat_tmp = mat_src.clone();
		for (int i = 0; i != n; ++i)
		{
			line(mat_tmp, pArr[i],pArr[(i+1)%n], color, 2);
		}
		imshow("img_with_line", mat_tmp);
		return 0;
	}
	int test_get_binary_gray_mat(const string& str_images_dir)
	{
		//获取指定目录中的所有图片
		vector<string> vec_files_path;
		sp_boost::get_files_path_list(str_images_dir, vec_files_path, ".jpg");
		for (int i = 0; i != vec_files_path.size(); ++i)
		{
			//加载图片
			Mat mat_src = imread(vec_files_path[i]);
			if (mat_src.empty())
			{
				printf("%s | read file:%s failed.", __FUNCTION__, vec_files_path[i].c_str());
				continue;
			}
			int nSize_blur = 7;
			blur(mat_src, mat_src, Size(nSize_blur, nSize_blur));
			imshow("src_mat", mat_src);
			//从原矩阵获取其对应的二值灰度矩阵
			Mat mat_src_gray;
			cv::cvtColor(mat_src, mat_src_gray, CV_BGR2GRAY);
			Mat mat_dst_binary;
			int nThreshold_binary = 110;
			cv::threshold(mat_src_gray, mat_dst_binary, nThreshold_binary, 255, THRESH_BINARY);
			cv::imshow("mat_dst", mat_dst_binary);
			cv::waitKey(0);
			//将结果写入文件
			boost::filesystem::path boost_path_src(vec_files_path[i]);
			string str_dst_path = string("F:\\GitHub\\OpenCVTest\\trunk\\OpencvTest\\images_result\\")
				+ boost_path_src.filename().string() + "_result.jpg";
			boost::filesystem::path boost_path_dst();
			cv::imwrite(str_dst_path, mat_dst_binary);
		}
		return 0;
	}
	int test_dilate(const string& str_img_path)
	{
		Mat img = imread(str_img_path);
		namedWindow("原始图", WINDOW_NORMAL);
		imshow("原始图", img);
		Mat out;
		//获取自定义核
		Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
		//膨胀操作
		dilate(img, out, element);
		namedWindow("膨胀操作", WINDOW_NORMAL);
		imshow("膨胀操作", out);
		waitKey(0);
		imwrite("./result_dilate.jpg", out);
		return 0;
	}
	int test_shrink(const string& str_img_path)
	{
		//		return test_resize(str_img_path);
		int nResize_height = 200;
		int nResize_width = 200;

		Mat mat_src = imread(str_img_path, IMREAD_GRAYSCALE);
		if (mat_src.empty())
		{
			printf("%s | fail to read image, path:%s.", __FUNCTION__, str_img_path.c_str());
			return -1;
		}
		imshow("src", mat_src);
		Mat mat_dst;
		//这两个方法需要src为彩色图
		//		scalePartAverage(mat_src, mat_dst, nResize_width * 1.0 / mat_src.cols, nResize_height * 1.0 / mat_src.rows);
		//		scaleIntervalSampling(mat_src, mat_dst, nResize_width * 1.0 / mat_src.cols, nResize_height * 1.0 / mat_src.rows);
		shrink_by_part_average(mat_src, mat_dst, nResize_width * 1.0 / mat_src.cols, nResize_height * 1.0 / mat_src.rows);
		imshow("dst", mat_dst);
		waitKey(0);
		return 0;
	}
	//缩放图像
	int test_resize(const string& str_img_path)
	{
		int nResize_height = 200;
		int nResize_width = 200;
		Mat mat_src = imread(str_img_path);
		if (mat_src.empty())
		{
			printf("%s | fail to read image, path:%s.", __FUNCTION__, str_img_path.c_str());
			return -1;
		}
		imshow("src", mat_src);
		Mat mat_dst;
		resize(mat_src, mat_dst, cv::Size(nResize_width, nResize_height), (0,0), (0, 0), cv::INTER_NEAREST);
		imshow("dst", mat_dst);
		waitKey(0);
		return 0;
	}
	void scaleIntervalSampling(const Mat &src, Mat &dst, double xRatio, double yRatio)
	{
		//只处理uchar型的像素
		CV_Assert(src.depth() == CV_8U);

		// 计算缩小后图像的大小
		//没有四舍五入，防止对原图像采样时越过图像边界
		int rows = static_cast<int>(src.rows * yRatio);
		int cols = static_cast<int>(src.cols * xRatio);

		dst.create(rows, cols, src.type());

		const int channesl = src.channels();

		switch (channesl)
		{
		case 1: //单通道图像
			{
				uchar *p;
				const uchar *origal;

				for (int i = 0; i < rows; i++){
					p = dst.ptr<uchar>(i);
					//四舍五入
					//+1 和 -1 是因为Mat中的像素是从0开始计数的
					int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
					origal = src.ptr<uchar>(row);
					for (int j = 0; j < cols; j++){
						int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
						p[j] = origal[col];  //取得采样像素
					}
				}
				break;
			}

		case 3://三通道图像
			{
				Vec3b *p;
				const Vec3b *origal;

				for (int i = 0; i < rows; i++) {
					p = dst.ptr<Vec3b>(i);
					int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
					origal = src.ptr<Vec3b>(row);
					for (int j = 0; j < cols; j++){
						int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
						p[j] = origal[col]; //取得采样像素
					}
				}
				break;
			}
		}
	}
	//基于局部均值的图像缩小
	void scalePartAverage(const Mat &src, Mat &dst, double xRatio, double yRatio)
	{
		int rows = static_cast<int>(src.rows * yRatio);
		int cols = static_cast<int>(src.cols * xRatio);

		dst.create(rows, cols, src.type());

		int lastRow = 0;
		int lastCol = 0;

		Vec3b *p;
		for (int i = 0; i < rows; i++) {
			p = dst.ptr<Vec3b>(i);
			int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;

			for (int j = 0; j < cols; j++) {
				int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;

				Vec3b pix;
				average(src, Point_<int>(lastRow, lastCol), Point_<int>(row, col), pix);
				p[j] = pix;

				lastCol = col + 1; //下一个子块左上角的列坐标，行坐标不变
			}
			lastCol = 0; //子块的左上角列坐标，从0开始
			lastRow = row + 1; //子块的左上角行坐标
		}
	}
	void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p)
	{

		const Vec3b *pix;
		Vec3i temp;
		for (int i = a.x; i <= b.x; i++){
			pix = img.ptr<Vec3b>(i);
			for (int j = a.y; j <= b.y; j++){
				temp[0] += pix[j][0];
				temp[1] += pix[j][1];
				temp[2] += pix[j][2];
			}
		}

		int count = (b.x - a.x + 1) * (b.y - a.y + 1);
		p[0] = temp[0] / count;
		p[1] = temp[1] / count;
		p[2] = temp[2] / count;
	}

	void shrink_by_part_average(const Mat &src, Mat &dst, double xRatio, double yRatio)
	{
		//只处理uchar型的像素
		CV_Assert(src.depth() == CV_8U);
		int rows = static_cast<int>(src.rows * yRatio);
		int cols = static_cast<int>(src.cols * xRatio);

		dst.create(rows, cols, src.type());
		int lastRow = 0;
		int lastCol = 0;
		if (src.channels() == 1)
		{//单通道
			uchar *p = NULL;
			for (int i = 0; i < rows; ++i) 
			{
				p = dst.ptr<uchar>(i);
				int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;

				for (int j = 0; j < cols; ++j) 
				{
					int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;

					uchar pix;
					average_for_1_channel(src, Point(lastCol, lastRow), Point(col, row), pix);
					p[j] = pix;
					lastCol = col + 1; //下一个子块左上角的列坐标，行坐标不变
				}
				lastCol = 0; //子块的左上角列坐标，从0开始
				lastRow = row + 1; //子块的左上角行坐标
			}
		}
		else if (src.channels() == 3)
		{//三通道
			Vec3b *p = NULL;
			for (int i = 0; i < rows; ++i) 
			{
				p = dst.ptr<Vec3b>(i);
				int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;

				for (int j = 0; j < cols; ++j) 
				{
					int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;

					Vec3b pix;
					average_for_3_channel(src, Point(lastCol, lastRow), Point(col, row), pix);
					p[j] = pix;
					lastCol = col + 1; //下一个子块左上角的列坐标，行坐标不变
				}
				lastCol = 0; //子块的左上角列坐标，从0开始
				lastRow = row + 1; //子块的左上角行坐标
			}
		}
	}

	void average_for_3_channel(const Mat &img, const Point& a, const Point& b, Vec3b &p)
	{

		const Vec3b *pix;
		Vec3i temp;
		for (int i = a.y; i <= b.y; ++i)
		{
			pix = img.ptr<Vec3b>(i);
			for (int j = a.x; j <= b.x; ++j)
			{
				temp[0] += pix[j][0];
				temp[1] += pix[j][1];
				temp[2] += pix[j][2];
			}
		}

		int count = (b.x - a.x + 1) * (b.y - a.y + 1);
		p[0] = temp[0] / count;
		p[1] = temp[1] / count;
		p[2] = temp[2] / count;
	}
	void average_for_1_channel(const Mat &img, const Point& a, const Point& b, uchar &nAverage)
	{
		CV_Assert(a.x <= b.x && a.y <= b.y);
		//所有像素之和
		int nSum = 0;
		for (int i = a.y; i <= b.y; ++i)
		{
			//行指针
			const uchar* pix = img.ptr<uchar>(i);
			for (int j = a.x; j <= b.x; ++j)
			{
				nSum += pix[j];
			}
		}
		//总元素个数
		int nCount = (b.x - a.x + 1) * (b.y - a.y + 1);
		nAverage = nSum / nCount;
	}
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
	int detect_circles(const Mat& src_bgr_mat, const Rect& roi, double dMin_centers_dist, vector<Vec3f>& circles, double dMax_canny_threshold = 100, double dCircle_center_threshold = 100,int minRadius = 0, int maxRadius = 0)
	{
		try
		{
			Mat mat_gray;
			/// Convert it to gray
			cvtColor( src_bgr_mat, mat_gray, CV_BGR2GRAY );
			namedWindow("gray image", 1);
			imshow("gray image", mat_gray);
			waitKey(50000);
			/// Reduce the noise so we avoid false circle detection
			GaussianBlur( mat_gray, mat_gray, Size(9, 9), 2, 2 );
			/// Apply the Hough Transform to find the circles
			HoughCircles( mat_gray, circles, CV_HOUGH_GRADIENT, 1, dMin_centers_dist, dMax_canny_threshold, dCircle_center_threshold, minRadius, maxRadius);
			if (circles.empty())
			{
				std::cout << "func:" << __FUNCTION__ << " | Can not find circles in mat" << std::endl; 
				return 10106;
			}
#ifdef _DEBUG
			/// Draw the circles detected
			for( size_t i = 0; i < circles.size(); i++ )
			{
				Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				int radius = cvRound(circles[i][2]);
				// circle center
				circle( src_bgr_mat, center, 3, Scalar(255, 255, 255), -1, 8, 0 );
				std::cout << "func:" << __FUNCTION__ << ", center:[" << i << "]" << Point(center.x + roi.x, center.y + roi.y) << std::endl;
				// circle outline
				circle( src_bgr_mat, center, radius, Scalar(0, 0, 255), 3, 8, 0 );
			}
			/// Show your results
			namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
			imshow( "Hough Circle Transform Demo", src_bgr_mat);
			waitKey(0);
#endif
			return 0;
		}
		catch (std::exception& e)
		{
			printf("%s | has exception, reason:%s", __FUNCTION__, e.what());
			return -1;
		}
	}
	int test_detect_circles(const std::string& str_img_path)
	{
		Mat src_mat = imread(str_img_path, IMREAD_COLOR);
		double dMin_centers_dist = src_mat.rows / 8;
		vector<Vec3f> circles;
		double dMax_canny_threshold = 50;
		double dCircle_center_threshold = 1;
		int minRadius = 0;
		int maxRadius = 0;

		int minX = 1381.219970703125;
		int minY = 607.6858520507812;
		int maxX = 1439.868408203125;
		int maxY = 682.8136596679688;
		Rect roi(minX, minY, maxX - minX, maxY - minY);
		Mat mat_sub(src_mat, roi);
		return detect_circles(mat_sub, roi, dMin_centers_dist, circles, dMax_canny_threshold, dCircle_center_threshold, minRadius, maxRadius);

	}
	int test_transform()
	{
		string str_kernel_err_reason;
		//在before.jpg中检测到的点A、B、C和00的坐标 （圆心）
		cv::Point point_A_before, point_B_before, point_C_before, point_00_before;
		point_A_before = Point(1411, 645);
		point_B_before = Point(1655, 1226);
		point_C_before = Point(1343, 1188);
		point_00_before = Point(1843, 257);
		//在after.jpg中检测到的点A、B、C和00的坐标 （圆心）
		cv::Point point_A_after, point_B_after, point_C_after, point_00_after;
		point_A_after = Point(1612, 776);
		point_B_after = Point(1091, 1131);
		point_C_after = Point(1067, 816);
		point_00_after = Point(1843, 258);
		//三角形纸片中斜边AB的长度，单位mm
		float fAB_len = 38.25; //TODO
		//后面此值通过外面传入。
		float fScale_pic_devide_mm_before = get_euclidean(point_A_before, point_B_before) / fAB_len;
		float fScale_pic_devide_mm_after = get_euclidean(point_A_after, point_B_after) / fAB_len;
		std::cout << __FUNCTION__ << " | scale of before:" << fScale_pic_devide_mm_before << std::endl;
		std::cout << __FUNCTION__ << " | scale of after:" << fScale_pic_devide_mm_after << std::endl;
		//打印三角形信息
		double dBias_degree = 0.5;
		bool bIs_right_triangle = is_special_right_triangle(point_A_before, point_B_before, point_C_before, dBias_degree);
		if (false == bIs_right_triangle)
		{
			std::cout << __FUNCTION__ << " | ERROR, before is not right_triangle" << std::endl;
		}
		bIs_right_triangle = is_special_right_triangle(point_A_after, point_B_after, point_C_after, dBias_degree);
		if (false == bIs_right_triangle)
		{
			std::cout << __FUNCTION__ << " | ERROR, after is not right_triangle" << std::endl;
		}
		//获取旋转变换矩阵
		cv::Point2f points_arr_before_Tri[3] = {point_A_before, point_B_before, point_C_before}; //
		cv::Point2f points_arr_after_Trie[3] = {point_A_after, point_B_after, point_C_after};
		cv::Mat warp_mat = cv::getAffineTransform(points_arr_before_Tri, points_arr_after_Trie);
		std::cout << __FUNCTION__ << ", warp_mat" << warp_mat << std::endl;
		//计算NC代码中的某点
		float fLast_X =-24.683/*-26.4255*/, fLast_Y = -22.824/*-23.6664*/; //TODO
		//对fLast_X, fLast_Y进行处理
		Point2f point_dst_to_nc;
		int ret = deal_nc_point(point_00_before, point_00_after, Point2f(fLast_X, fLast_Y), warp_mat
			, fScale_pic_devide_mm_before, fScale_pic_devide_mm_after, point_dst_to_nc, str_kernel_err_reason);
		//将转换所得结果放入结果串中
		float fT_X = point_dst_to_nc.x; //转换后的值
		float fT_Y = point_dst_to_nc.y;
		std::cout <<__FUNCTION__ << " | In NC, result x:" << fT_X << ", y:" << fT_Y << std::endl;
		return 0;
	}
	double get_euclidean(const cv::Point& pointO, const cv::Point& pointA)
	{
		double distance;  
		distance = powf((pointO.x - pointA.x),2) + powf((pointO.y - pointA.y),2);  
		distance = sqrtf(distance);  

		return distance;
	}
	int deal_nc_point(const Point& point_00_before, const Point& point_00_after
		, const Point2f& point_src_from_nc, const cv::Mat& warp_mat, const float& fScale_pic_devide_mm_before
		, const float& fScale_pic_devide_mm_after, Point2f& point_dst_to_nc, std::string& str_kernel_err_reason)
	{
		//nc代码中的点结合00点，得到其在原图中的绝对坐标
		Point2f point_src_in_img; //点在原图中的绝对坐标
		point_src_in_img.x = point_00_before.x + point_src_from_nc.x * fScale_pic_devide_mm_before;
		point_src_in_img.y = point_00_before.y - point_src_from_nc.y * fScale_pic_devide_mm_before;
		point_dst_to_nc = get_dst_point_after_affine(point_src_in_img, warp_mat);
		//转换为以00点为原点的相对坐标
		point_dst_to_nc.x = (point_dst_to_nc.x - point_00_after.x) / fScale_pic_devide_mm_after;
		point_dst_to_nc.y = (point_00_after.y - point_dst_to_nc.y) / fScale_pic_devide_mm_after;
		return 0;
	}

	cv::Point2f get_dst_point_after_affine(const cv::Point2f& src_point, const Mat& affine_transform_mat)
	{
		cv::Point2f dst_point;
		if (affine_transform_mat.type() == CV_32FC1)
		{
			dst_point.x = affine_transform_mat.at<float>(0, 0) * src_point.x + affine_transform_mat.at<float>(0, 1) * src_point.y + affine_transform_mat.at<float>(0, 2);
			dst_point.y = affine_transform_mat.at<float>(1, 0) * src_point.x + affine_transform_mat.at<float>(1, 1) * src_point.y + affine_transform_mat.at<float>(1, 2);
		}
		else if (affine_transform_mat.type() == CV_64FC1)
		{
			dst_point.x = affine_transform_mat.at<double>(0, 0) * src_point.x + affine_transform_mat.at<double>(0, 1) * src_point.y + affine_transform_mat.at<double>(0, 2);
			dst_point.y = affine_transform_mat.at<double>(1, 0) * src_point.x + affine_transform_mat.at<double>(1, 1) * src_point.y + affine_transform_mat.at<double>(1, 2);
		}
		else
		{
			printf("func:%s | type of affine mat is %d, not equal to CV_32FC1:%d, CV_64FC1:%d"
				, __FUNCTION__, affine_transform_mat.type(), CV_32FC1, CV_64FC1);
		}
		return dst_point;
	}

	bool is_special_right_triangle(const cv::Point& point_A, const cv::Point& point_B
		, const cv::Point& point_C, const double& dBias_degree)
	{
		double dDistance_AB = get_euclidean(point_A, point_B);
		double dDisstance_BC = get_euclidean(point_B, point_C);
		double dDistance_AC = get_euclidean(point_A, point_C);
		//两边之和大于第三边
		if (dDisstance_BC + dDistance_AC <= dDistance_AB)
		{
			printf("func:%s | AC + BC <= AB.\n", __FUNCTION__);
			return false;
		}

		double dRatio_AC_AB = dDistance_AC / dDistance_AB;
		double dRatio_BC_AB = dDisstance_BC / dDistance_AB;
		//夹角度数
		double dDegree_BAC =  acos(dDistance_AC / dDistance_AB) * 180 /CV_PI;
		double dDegree_ABC = acos(dDisstance_BC / dDistance_AB) * 180 /CV_PI;
#ifdef _DEBUG
		std::cout << "func:" << __FUNCTION__ << ", point A:" << point_A << ", point B:" << point_B << ", point C:" << point_C << std::endl;
		std::cout << "\t" <<"dDistance_AB:" << dDistance_AB << ", dDisstance_BC:" << dDisstance_BC << ", dDistance_AC:" << dDistance_AC << std::endl;
		std::cout << "\t" << "dRatio_AC_AB:" << dRatio_AC_AB << ",dRatio_BC_AB:" << dRatio_BC_AB << std::endl;
		std::cout << "\t" << "degree of BAC:" << dDegree_BAC << ", degree of ABC:" << dDegree_ABC << std::endl;
#endif
		if (fabs(dDegree_BAC - 30/*36.87*/) < dBias_degree && fabs(dDegree_ABC - 60/*53.13*/) < dBias_degree)
		{//角度满足关系
			return true;
		}

		printf("func:%s | AC/AB:%f, BC/AB:%f\n", __FUNCTION__, dRatio_AC_AB, dRatio_BC_AB);
		return false;
	}
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
	int change_contrast_and_brightness(const Mat& mat_src_bgr, double udContrast, int nBrightness, Mat& mat_dst_image)
	{
		//判定参数合法性
		if (mat_src_bgr.empty())
		{//原图为空
			printf("%s | src mat is empty", __FUNCTION__);
			return -1;
		}
		if (udContrast <= 0)
		{
			printf("%s | contrast must be more than 0", __FUNCTION__);
			return 10106;
		}
		mat_dst_image = Mat::zeros( mat_src_bgr.size(), mat_src_bgr.type());

		/// 执行运算 mat_dst_image(i,j) = udContrast*mat_src_bgr(i,j) + nBrightness
		for( int y = 0; y < mat_src_bgr.rows; y++ )
		{
			for( int x = 0; x < mat_src_bgr.cols; x++ )
			{
				for( int c = 0; c < 3; c++ )
				{
					mat_dst_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( udContrast*( mat_src_bgr.at<Vec3b>(y,x)[c] ) + nBrightness );
				}
			}
		}
#ifdef _DEBUG
		/// 创建窗口
		namedWindow("Original Image", 1);
		namedWindow("New Image", 1);

		/// 显示图像
		imshow("Original Image", mat_src_bgr);
		imshow("New Image", mat_dst_image);

		/// 等待用户按键
		waitKey(5000);
#endif
		return 0;

	}
	static void callback_change_contrast_and_brightness(int pos, void* userdata)
	{
		printf("%s | enter\n", __FUNCTION__);
		string* pStr = reinterpret_cast<string* >(userdata);
		Mat mat_src = imread(*pStr, IMREAD_COLOR);
		static int nContrast = 1;
		static int nBrightness = 13;
		if (pos <= 10)
		{
			nContrast = pos;
		}
		else
		{
			nBrightness = pos;
		}
		std::cout << __FUNCTION__ << " | nContrast:" << nContrast << ", nBrightness:" << nBrightness <<  ", pos:" << pos << std::endl;
		Mat mat_dst;
		int ret =  CBusin_OpenCV_Common_Tool::instance().change_contrast_and_brightness(mat_src, nContrast, nBrightness, mat_dst);
		if (ret)
		{
			printf("%s | error, ret:%d", __FUNCTION__, ret);
		}
	}
	int test_change_contrast_and_brightness(const string& str_img_path)
	{
		int nContrast = 1;
		int nBrightness = 13; 
		/// 创建显示窗口
		char* window_name = "My Test";
		namedWindow( window_name, CV_WINDOW_AUTOSIZE );
		createTrackbar("contrast threshold", window_name, &nContrast, 10, callback_change_contrast_and_brightness, (void*)&str_img_path);
		createTrackbar("brightness threshold", window_name, &nBrightness, 100, callback_change_contrast_and_brightness, (void*)&str_img_path);

		callback_change_contrast_and_brightness(nContrast, (void*)&str_img_path);
		waitKey();
		return 0;
	}
	int test_wenzi_G_code(const string& str_img_path)
	{
		try
		{
			/// 装载图像
			const Mat mat_src = imread(str_img_path);
			if( !mat_src.data )
			{ 
				return -1; 
			}
			Mat mat_dst_binary;
			Point point_00(4258, 232);
			int nThreshold_binary = 110; //120
			int ret = get_binary_gray_mat(mat_src, 110, mat_dst_binary);
			if (ret != 0)
			{
				std::cout << __FUNCTION__ << " | get binary mat failed, ret:" << ret << endl; 
				return ret;
			}
			imwrite("./wenzi_result.jpg", mat_dst_binary);
			namedWindow("mat_for_nc", WINDOW_NORMAL | WINDOW_KEEPRATIO);
			imshow("mat_for_nc", mat_dst_binary);
			waitKey(0);

			float fScale_pic_devide_mm = 244.10 / 6.73;
			string str_nc_data;
			float fZ_up = 1;
			float fZ_down = -0.5;
			ret = create_nc_code(mat_dst_binary, point_00, fScale_pic_devide_mm, fZ_up, fZ_down, str_nc_data);
			if (ret)
			{
				std::cout << __FUNCTION__ << " | error line:" << __LINE__ << endl;
				return ret;
			}
			ofstream fout("final_result.nc");
			fout << str_nc_data;
			return 0;
		}
		catch (std::exception& e)
		{
			std::cout << "fun:" << __FUNCTION__ << ", error reason:" << e.what() << std::endl;
			return -1;
		}
	}
	int test_shrink_mat()
	{
		/// 装载图像
		const Mat mat_src = imread(/*"wenzi.jpg"*/"./wenzi_zhongguo.jpg");
		if( !mat_src.data )
		{ 
			return -1; 
		}
		//去除噪声
		int nSize_blur = 7;
		blur(mat_src, mat_src, Size(nSize_blur, nSize_blur));
		//threshold(image,image,0,255,CV_THRESH_OTSU);
		Mat mat_dst_binary_gray;
		int nThreshold_binary = 110; //120 //此值越大，说明越不容易被判定为白色而越容易被判定为黑色
		int ret = get_binary_gray_mat(mat_src, nThreshold_binary, mat_dst_binary_gray);
		if (ret != 0)
		{
			std::cout << __FUNCTION__ << " | get binary mat failed, ret:" << ret << endl; 
			return ret;
		}
		//将所得的二值均值中的黑点进行过滤一下，去除误判的黑点
		//		remove_error_black_points(mat_dst_binary);
		imwrite("./wenzi_result2.jpg", mat_dst_binary_gray);
		//		std::cout << __FUNCTION__ << " | mat_dst_binary:" << mat_dst_binary << endl;
		std::cout << __FUNCTION__ << " | CV_8UC1:" << CV_8UC1 << endl;
		std::cout << __FUNCTION__ << " | mat_dst_binary type:" << mat_dst_binary_gray.type() << endl;
		std::cout << __FUNCTION__ << " | mat_dst_binary channel:" << mat_dst_binary_gray.channels() << endl;
		//将二值矩阵按指定大小进行缩放
		float fScale_pic_devide_mm = 0;
		float fMax_width_pxl = 0;
		float fMax_height_pxl = 0;
		Point point_00;
#if 0 //你
		fScale_pic_devide_mm = 244.10 / 6.73;
		fMax_width_pxl = 8.5 * fScale_pic_devide_mm;
		fMax_height_pxl = 9.0 * fScale_pic_devide_mm;
		point_00 = Point(4258, 232);
#else //中国
		fScale_pic_devide_mm = 685.29 / 14.27;
		fMax_width_pxl = 15 * fScale_pic_devide_mm;
		fMax_height_pxl = 12 * fScale_pic_devide_mm;
		point_00 = Point(3096, 288);
#endif	
		Mat mat_binary_shrinked;
		ret = rotate_and_shrink_binary_writting_mat(mat_dst_binary_gray, fMax_width_pxl, fMax_height_pxl, fScale_pic_devide_mm, point_00, mat_binary_shrinked);

		if (ret)
		{
			std::cout << __FUNCTION__ << " | shrink binary mat failed, ret:" << ret << endl;
			return ret;
		}
		imwrite("./weinzi_shinked_result.jpg", mat_binary_shrinked);
		// 		namedWindow("mat_for_nc", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		// 		imshow("mat_for_nc", mat_binary_shrinked);
		// 		waitKey(0);
		blur(mat_binary_shrinked, mat_binary_shrinked, Size(nSize_blur, nSize_blur));
		//对结果再次进行二值化
		Mat mat_binary_for_NC;
		cv::threshold(mat_binary_shrinked, mat_binary_for_NC, nThreshold_binary, 255, THRESH_BINARY);
		string str_nc_data;
		namedWindow("mat_for_nc", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("mat_for_nc", mat_binary_for_NC);
		waitKey(0);
		float fZ_up = 1;
		float fZ_down = -0.5;
		ret = create_nc_code(mat_binary_for_NC, point_00, fScale_pic_devide_mm, fZ_up, fZ_down, str_nc_data);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | error line:" << __LINE__ << endl;
			return ret;
		}
		ofstream fout("final_result_after_affine.nc");
		fout << str_nc_data;
		return 0;
	}
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
	int get_binary_gray_mat(const cv::Mat& mat_src, int nThreshold_binary, Mat& mat_dst_binary)
	{
		Mat mat_src_gray;
		// 		/// 原图像转换为灰度图像
		cvtColor( mat_src, mat_src_gray, CV_BGR2GRAY );
		//使用阈值法:大于阈值，则为白色（255），否则为黑色（0）
		cv::threshold(mat_src_gray, mat_dst_binary, nThreshold_binary, 255, THRESH_BINARY);
		if (mat_dst_binary.type() != CV_8UC1)
		{
			std::cout << __FUNCTION__ << " | type of mat is:" << mat_dst_binary.type() << ", not CV_8UC1:" << CV_8UC1 << endl;
			return -1;
		}
		return 0;
	}
	int create_nc_code(const Mat& mat_dst_binary, const Point& point_00 
		, float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data)
	{
		size_t nLastN = 0;
		str_nc_data += get_nc_head(nLastN);
		int ret = 0;
		ret = get_nc_effective_data(mat_dst_binary, point_00, nLastN, fScale_pic_devide_mm, fZ_up, fZ_down, str_nc_data);
		str_nc_data += get_nc_tail(nLastN);
		return 0;
	}
	int get_nc_effective_data(const Mat& mat_binary, const Point& point_00, size_t& nLastN , float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data)
	{
		bool bIs_first_black = false; //第一个黑点为第一行有效数字，即N9
		bool bIs_lifting_knife = true; //遍历某列时,雕刻刀的状态，如果雕刻刀是抬刀状态，则在雕刻某点事，则在此点应该先抬刀再雕刻。
		string str_slowdown_flg = "F100";
		for (int i = 0; i != mat_binary.rows; ++i)
		{
			for (int j = 0; j != mat_binary.cols; ++j)
			{
				//判定当前点是否为黑色
				if (mat_binary.at<uchar>(i, j) - 0 < 10)
				{//当前点很可能为黑色
					Point2f point_dst_to_nc;
					//转换为以00点为原点的相对坐标,单位mm
					point_dst_to_nc.x = (j - point_00.x) / fScale_pic_devide_mm;
					point_dst_to_nc.y = (point_00.y - i) / fScale_pic_devide_mm;
					if (false == bIs_first_black)
					{//当前黑点为全局第一个遇到的黑点
						bIs_first_black = true;
						//抬刀
						string str_N9 = string("N") + boost::lexical_cast<string>(nLastN++) + " G0X" + boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
						str_nc_data += str_N9 + "\n";
						//缓慢下刀雕刻
						string str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + str_slowdown_flg;
						str_nc_data += str_row + "\n";
						bIs_lifting_knife = false;
					}
					else
					{//当前黑点非全局第一个遇到的黑点
						//雕刻当前黑点
						string str_row;
						//判定之前刀是否为抬刀状态
						if (bIs_lifting_knife)
						{//之前处于抬刀状态，则在此点先抬刀
							str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
								+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
							str_nc_data += str_row + "\n";
							//再降速雕刻
							str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
								+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + str_slowdown_flg;
							str_nc_data += str_row + "\n";
						}
						str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down);
						str_nc_data += str_row + "\n";
						bIs_lifting_knife = false; //设置刀为非抬刀状态
					}
					//判定下一个点存在，则判定其是否为白色，如果为白色，则需要添加抬刀代码
					if (j + 1 != mat_binary.cols && 255 - mat_binary.at<uchar>(i, j + 1) < 10)
					{//下一个点未越界且为白色,则刀需要抬起
						//设置行以使得刀抬起
						string str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
						str_nc_data += str_row + "\n";
						bIs_lifting_knife = true;
					}
				}
			}
		}
		return 0;
	}
	/************************************
	* Method:    get_nc_head
	* Brief:  获取文件头及下一行位置
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: size_t & nLastN -[in/out]  
	************************************/
	string get_nc_head(size_t& nLastN)
	{
		string str_head_flg = string("N1 G90 G53 Z0\n")
			+ "N2 S24000M03\n"
			+ "N3 M08\n"
			+ "N4 G04 X3.\n"
			+ "N5 G17 G90 G01 G54\n"
			+ "N6 G00 X0Y0\n"
			+ "N7 ([锥度平底]JD-45-0.10)\n"
			+ "N8 G17\n";
		nLastN = 9; //文件头总函数+1
		return str_head_flg;
	}
	/************************************
	* Method:    get_nc_tail
	* Brief:  根据当前行下标来构造NC代码的尾部
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: const size_t & nLastN -[in] 当前总行数+1  
	************************************/
	string get_nc_tail(const size_t& nLastN)
	{
		size_t nTmp_lastN = nLastN;
		string str_tail_flg = string("N") + boost::lexical_cast<string>(nTmp_lastN++) + string(" G0Z5.000\n");
		str_tail_flg += string("N") + boost::lexical_cast<string>(nTmp_lastN++) + string(" G90 G53 Z0\n");
		str_tail_flg += string("N") + boost::lexical_cast<string>(nTmp_lastN++) + string(" M05\n");
		str_tail_flg += string("N") + boost::lexical_cast<string>(nTmp_lastN++) + string(" M09\n");
		str_tail_flg += string("N") + boost::lexical_cast<string>(nTmp_lastN++) + string(" Y0\n");
		str_tail_flg += string("N") + boost::lexical_cast<string>(nTmp_lastN++) + string(" M30\n");
		// 		string("N242413 G0Z5.000\n")
		// 			+ "N242414 G90 G53 Z0\n"
		// 			+ "N242415 M05\n"
		// 			+ "N242416 M09\n"
		// 			+ "N242417 Y0\n"
		// 			+ "N242418 M30\n";
		return str_tail_flg;
	}
	int rotate_and_shrink_binary_writting_mat(const Mat& mat_src_binary_gray, const float& fMax_width_pxl
		, const float& fMax_height_pxl, const float& fScale_pic_devide_mm, const Point& point_00, Mat& mat_binary_gray_shrinked)
	{
		int ret = 0;
		mat_binary_gray_shrinked = Mat(mat_src_binary_gray.size(), mat_src_binary_gray.type(), Scalar(255));
		//遍历二值矩阵，此时要求文字处为黑色，其他均为白色。
		vector<Point> vec_writting_points; //文字对应的点列表
		int  nCircle_radius = 8; //数值要大于黄色圆半径，单位mm
		int n00_threshed = nCircle_radius * fScale_pic_devide_mm;
		for (size_t nRow = 0; nRow != mat_src_binary_gray.rows; ++nRow)
		{
			for (size_t nCol = 0; nCol != mat_src_binary_gray.cols; ++nCol)
			{
				//跳过00点到右上角这块区域,避免将00点也误作为字体了
				if (nCol > point_00.x - n00_threshed && nRow < point_00.y + n00_threshed)
				{
					continue;
				}
				//判定是否为黑色点
				if (mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
				{
					vec_writting_points.push_back(Point(nCol, nRow));
				}
			}
		}
		if (vec_writting_points.empty())
		{
			std::cout << __FUNCTION__ << " | error, vec is empty" << endl;
			return -1;
		}
		//显示原图
		//		imshow("mat_src_binary", mat_src_binary);
		//查找点集的最小外接矩形
		RotatedRect rect_points = minAreaRect(vec_writting_points);
		//定义一个存储以上四个点的坐标的变量
		Point2f points_rect_list[4]; //The order is bottomLeft, topLeft, topRight, bottomRight
		ret = get_rect_points(mat_src_binary_gray, rect_points, points_rect_list, 4);
		if (ret != 0)
		{
			std::cout << __FUNCTION__ << " | error, line:" << __LINE__ << endl;
			return ret;
		}
		//画出矩形
		draw_lines(mat_src_binary_gray, points_rect_list, 4, "./img_with_line.jpg");
		//求取最小缩放比例
		float fHeight_pxl = get_euclidean((Point)(points_rect_list[0]), (Point)(points_rect_list[1]));
		float fWidth_pxl = get_euclidean((Point)(points_rect_list[1]), (Point)(points_rect_list[2]));
		double fx = (fMax_width_pxl - 0.5 * fScale_pic_devide_mm) / fWidth_pxl; //离最大边界框预留一定距离
		double fy = (fMax_height_pxl - 0.5 * fScale_pic_devide_mm) / fHeight_pxl;
		double fScale = std::min(fx, fy); //以最小值作为整体的缩放比例，注意：x、y方向不可使用不同的缩放比例，否则变形
		//将矩形旋转至水平并进行缩放
		ret = rotate_writting_to_horizontal_direction(mat_src_binary_gray, rect_points, fScale, fScale_pic_devide_mm, mat_binary_gray_shrinked);
		if (ret)
		{
			std::cout << __FUNCTION__ << " |  error, line:" << __LINE__ << ", ret:" << ret << endl;
			return ret;
		}
		return 0;
	}
	int shrink_binary_mat_using_contours(const Mat& mat_src_binary, const float& fMax_width_pxl
		, const float& fMax_height_pxl, const Point& point_00, Mat& mat_binary_shrinked)
	{
		namedWindow("mat_src_binary", WINDOW_AUTOSIZE);
		imshow("mat_src_binary",mat_src_binary);
		waitKey(50000);
		//查找轮廓
		//寻找最外层轮廓
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		Mat mat_sub(mat_src_binary(Rect(0, 0, point_00.x - 200, mat_src_binary.rows)));
		findContours(mat_sub, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_NONE, Point());
		Mat imageContours=Mat(mat_sub.size(),CV_8UC1, Scalar(255));
		//查找最小外接矩形
		Point2f P[4];
		for (int i = 0; i != contours.size(); ++i)
		{
			//绘制轮廓
			//绘制轮廓
			drawContours(imageContours, contours, i, Scalar(0), 1, 8, hierarchy);
			//绘制轮廓的最小外结矩形
			RotatedRect rect=minAreaRect(contours[i]);
			rect.points(P);
		}
		namedWindow("MinAreaRect", WINDOW_AUTOSIZE);
		imshow("MinAreaRect",imageContours);
		imwrite("./contours_result.jpg", imageContours);
		waitKey(0);
		return 0;
	}
	Mat override_sub_mat(const Mat& m1, const Mat& m2, const Point& rio_loction_point)
	{
		Mat tmp_mat = m1.clone();
		//		CV_ASSERT(rio_loction_point.x + m2.cols < m1.cols && rio_loction_point.y + m2.rows < m1.rows );
		for (size_t idx_row = 0; idx_row != m2.rows && idx_row + rio_loction_point.y < m1.rows; ++idx_row)
		{
			for (size_t idx_col = 0; idx_col != m2.cols && idx_col + rio_loction_point.x < m1.cols; ++idx_col)
			{
				tmp_mat.at<Vec3b>(idx_row + rio_loction_point.y, idx_col + rio_loction_point.x) = m2.at<Vec3b>(idx_row, idx_col);
			}
		}
		return tmp_mat;
	}
	int  test_override_sub_mat()
	{
		//设置深蓝色背景画布
		cv::Mat image = cv::Mat::zeros(512, 512, CV_8UC3);
		image.setTo(cv::Scalar(100, 0, 0));
		cv::imshow("original", image);

		//读取待复制图片
		cv::Mat roi = cv::imread("E:\\Images\\Hepburn.png", cv::IMREAD_COLOR);
		cv::imshow("roi", roi);

		//设置画布绘制区域并复制
		cv::Rect roi_rect = cv::Rect(128, 128, roi.cols, roi.rows);
		roi.copyTo(image(roi_rect));

		cv::imshow("result", image);

		cv::waitKey(0);

		return 0;
	}
	int remove_error_black_points(Mat& mat_binary)
	{
		size_t nStep = 5;
		//遍历Mat中的每一个点，计算其周围的黑点数目，如果超过阈值，则认为其是字体的一部分
		size_t nThreshold_black_points_num = 3;
		for (size_t nRow = 0; nRow != mat_binary.rows; ++nRow)
		{
			for (size_t nCol = 0; nCol != mat_binary.cols; ++nCol)
			{
				size_t nBlack_points_num = calc_black_points_num(mat_binary, Point(nCol, nRow), nStep);
				if (nBlack_points_num < nThreshold_black_points_num)
				{
					mat_binary.at<uchar>(nRow, nCol) = 255;
				}
				//				std::cout << __FUNCTION__ << " | row:" << nRow << ", col:" << nCol << endl;
			}
		}
		return 0;
	}
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
	int calc_black_points_num(const Mat& mat_binary, const Point& point_current, size_t nStep)
	{
		size_t nBlack_points_num = 0;
		size_t nMin_x = point_current.x - (int)nStep >=0 ? point_current.x - nStep : 0;
		size_t nMin_y = point_current.y - (int)nStep >= 0 ? point_current.y - nStep : 0;
		size_t nMax_x = point_current.x + nStep < mat_binary.cols ? point_current.x + nStep : mat_binary.cols;
		size_t nMax_y = point_current.y + nStep < mat_binary.rows ? point_current.y + nStep : mat_binary.rows;
		for (size_t nY = nMin_y; nY != nMax_y; ++nY)
		{
			for (size_t nX = nMin_x; nX != nMax_x; ++nX)
			{
				if (mat_binary.at<uchar>(nY, nX) < 10)
				{//是黑色
					++nBlack_points_num;
				}
			}
		}
		return nBlack_points_num;
	}
	int draw_lines(const Mat& mat_src, Point2f* pArr, size_t n, const string& str_img_path)
	{
		Mat mat_tmp = mat_src.clone();
		for (int i = 0; i != n; ++i)
		{
			line(mat_tmp, pArr[i],pArr[(i+1)%n],Scalar(0), 2);
		}
		imwrite(str_img_path, mat_tmp);
		return 0;
	}

	//将矩形旋转至水平方向
	int rotate_writting_to_horizontal_direction(const Mat& src_mat, const RotatedRect& rRect
		, const float& fScale, const float& fScale_pic_devide_mm, Mat& mat_dst)
	{
		//获取变换前的最小外接矩形的四个顶点
		Point2f src_rect_points_arr[4]; //The order is bottomLeft, topLeft, topRight, bottomRight
		int ret = get_rect_points(src_mat, rRect, src_rect_points_arr, 4);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | error, line:" << __LINE__ << endl;
			return ret;
		}
		//以外接矩形的中心作为旋转中心
		CvPoint2D32f  center = cvPoint2D32f(rRect.center.x, rRect.center.y);
		double Ddgree = 0; //为负的表示逆时针旋转,单位角度
		if (rRect.center.x < src_rect_points_arr[0].x)
		{
			Ddgree = -(90 + rRect.angle);
		}
		else
		{
			Ddgree = -rRect.angle;
		}
		Mat mat_rotated;//变换所得矩阵
		//为结果矩阵构造背景：纯白色背景，大小、类型与源矩阵大小一致
		mat_dst = Mat(src_mat.size(),src_mat.type(), Scalar(255));
		//获取仿射矩阵以及旋转后矩阵的大小
		Mat mat_rotation; //仿射矩阵
		Size Dsize;
		//根据旋转中心、缩放比例获取仿射变换矩阵和结果矩阵的大小
		ret = CBusin_Opencv_Transform_Tool::instance().get_rotation_matrix_without_loss(src_mat, center, Ddgree, fScale, mat_rotation, Dsize);
		if (ret)
		{
			std::cout << " | error, line:" << __LINE__ << ", ret:" << ret << endl;
			return ret;
		}
		//旋转变换，以摆正外接矩形，即将文字尽量摆正
		cv::warpAffine(src_mat, mat_rotated, mat_rotation, Dsize, 1, 0, 0);
		imwrite("./mat_rotated.jpg", mat_rotated);

		Point2f dst_rect_points_arr[4];
		//获取变换后最小外接矩形对于的四个顶点
		ret = get_points_after_affined(mat_rotation, src_rect_points_arr, 4, dst_rect_points_arr);
		//此时外接矩形是正的
		//将顶点列表进行排序，以使其顺序为bottomLeft, topLeft, topRight, bottomRight
		ret = sort_rect_points(dst_rect_points_arr, 4);
		//将外接矩形放大,避免遮挡了字体
		int nExt_size_pxl = 0.5 * fScale_pic_devide_mm; //四条边放大的大小，单位像素
		ret = increase_rect_size(dst_rect_points_arr, 4, nExt_size_pxl);
		//将旋转后的外接矩形画出来
		ret = draw_lines(mat_rotated, dst_rect_points_arr, 4, "./img_with_line_after_retated.jpg");
		//将文字对应的区域提取出来
		//获取文字区域对应的矩形
		size_t nWidth_roi = dst_rect_points_arr[2].x - dst_rect_points_arr[1].x /*rRect.size.width * fScale*/; //可能有问题
		size_t nHeight_roi = dst_rect_points_arr[3].y - dst_rect_points_arr[2].y /*rRect.size.height * fScale*/;
		Rect dst_rect_roi(dst_rect_points_arr[1].x, dst_rect_points_arr[1].y, nWidth_roi, nHeight_roi);
		//文字区域矩阵
		Mat mat_roi(mat_rotated(dst_rect_roi));
		imwrite("./mat_roi.jpg", mat_roi);
		//将提取出的文字区域覆盖到背景矩阵中，以得到最终结果矩阵
		//结果矩阵中的目标区域（即文字在结果矩阵中的位置）
		Rect src_rect_roi(rRect.center.x - rRect.size.width / 2, rRect.center.y - rRect.size.height / 2, nWidth_roi, nHeight_roi);
		mat_roi.copyTo(mat_dst(src_rect_roi));
		return 0;
	}

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
	int get_points_after_affined(const Mat& mat_rotation, const Point2f* pSrc_points, size_t nPoints_num, Point2f* pDst_points)
	{
		for (int i = 0; i != nPoints_num; ++i)
		{
			pDst_points[i] = get_dst_point_after_affine(pSrc_points[i], mat_rotation);
		}
		return 0;
	}
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
	int get_rect_points(const Mat& mat_src, const RotatedRect& rRect, Point2f* pts, size_t nPoints_num = 4)
	{
		//将rect_points变量中存储的坐标值放到pts数组中
		rRect.points(pts);
		//校验所得的点是否合法，如果存在非法值，则报错
		for (size_t i =0; i != nPoints_num; ++i)
		{
			if (pts[i].x < 0)
			{
				std::cout << __FUNCTION__ << " | invalid x:" << pts[i].x << endl;
				return -1;
			}
			if (pts[i].x >= mat_src.cols)
			{
				std::cout << __FUNCTION__ << " | invalid x:" << pts[i].x << endl;
				return -1;
			}
			if (pts[i].y < 0)
			{
				std::cout << __FUNCTION__ << " | invalid y:" << pts[i].y << endl;
				return -1;
			}
			if (pts[i].y >= mat_src.rows)
			{
				std::cout << __FUNCTION__ << " | invalid y:" << pts[i].y << endl;
				return -1;
			}
		}
		return 0;
	}

	/************************************
	* Method:    sort_rect_points
	* Brief:  对矩形的顶点列表进行排序，以使得其点顺序为bottomLeft, topLeft, topRight, bottomRight
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  作为输入时：矩形的顶点数组；作为输出时：四个顶点被排序了
	*Parameter: size_t nPoints_num -[in] 顶点个数
	************************************/
	int sort_rect_points(Point2f* pRect_points, size_t nPoints_num)
	{
		Point2f bottomLeft, topLeft, topRight, bottomRight;
		bottomLeft= topLeft	= topRight = bottomRight = pRect_points[0];
		float fThreshold = 0.01; //浮点数相等判定时的阈值
		for (size_t i = 0; i != nPoints_num; ++i)
		{
			if ((pRect_points[i].x < bottomLeft.x || fabs(pRect_points[i].x - bottomLeft.x) < fThreshold)
				&& (pRect_points[i].y > bottomLeft.y || fabs(pRect_points[i].y - bottomLeft.y) < fThreshold))
			{//bottomLeft的x最小且y最大
				bottomLeft = pRect_points[i];
			}
			if ((pRect_points[i].x < topLeft.x || fabs(pRect_points[i].x - topLeft.x) < fThreshold) 
				&& (pRect_points[i].y < topLeft.y || fabs(pRect_points[i].y - topLeft.y) < fThreshold))
			{//topLeft的x最小且y最小
				topLeft = pRect_points[i];
			}
			if ((pRect_points[i].x > topRight.x || fabs(pRect_points[i].x - topRight.x) <fThreshold) 
				&& (pRect_points[i].y < topRight.y || fabs(pRect_points[i].y - topRight.y ) < fThreshold))
			{//topRight的x最大且y最小
				topRight = pRect_points[i];
			}
			if ((pRect_points[i].x > bottomRight.x || fabs(pRect_points[i].x - bottomRight.x) < fThreshold)
				&& (pRect_points[i].y > bottomRight.y || fabs(pRect_points[i].y - bottomRight.y) < fThreshold))
			{//bottomRight的x最大且y最大
				bottomRight = pRect_points[i];
			}
		}
		pRect_points[0] = bottomLeft;
		pRect_points[1] = topLeft;
		pRect_points[2] = topRight;
		pRect_points[3] = bottomRight;
		return 0;
	}
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
	int increase_rect_size(Point2f* pRect_points, size_t nPoints_num, int nExt_size_pxl)
	{
		//bottomLeft的x变小且y变大
		pRect_points[0].x -= nExt_size_pxl;
		pRect_points[0].y += nExt_size_pxl;
		//topLeft的x变小且y变小
		pRect_points[1].x -= nExt_size_pxl;
		pRect_points[1].y -= nExt_size_pxl;
		//topRight的x变大且y变小
		pRect_points[2].x += nExt_size_pxl;
		pRect_points[2].y -= nExt_size_pxl;
		//bottomRight的x变大且y变大
		pRect_points[3].x += nExt_size_pxl;
		pRect_points[3].y += nExt_size_pxl;
		return 0;
	}
	void get_connected_contours(const Mat& mat_src_binary_gray, vector<vector<Point> >& vec_contours)
	{
		Mat mat_tmp_binary_gray = mat_src_binary_gray.clone();
		for (int nRow = 0; nRow != mat_tmp_binary_gray.rows; ++nRow)
		{
			for (int nCol = 0; nCol != mat_tmp_binary_gray.cols; ++nCol)
			{
				if (mat_tmp_binary_gray.at<uchar>(nRow, nCol) < 10)
				{//当前点为黑色
					vector<Point> vec_points;
					traverse_connected_graph(mat_tmp_binary_gray, Point(nCol, nRow), vec_points);
					vec_contours.push_back(vec_points);
				}
			}
		}
	}
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
	void traverse_connected_graph(Mat& mat_src_binary_gray, const Point& point_start, vector<Point>& vec_points)
	{
		int nRow = point_start.y, nCol = point_start.x;
		while (nRow != mat_src_binary_gray.rows  && nRow >= 0 && nCol >= 0 && nCol != mat_src_binary_gray.cols 
			&& mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
		{
			//一直往右走
			while (nCol != mat_src_binary_gray.cols && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow, nCol++) = 255;//用完后设置为白色
			}
			//此时遇到无效数据，往左退一格，并往下走
			--nCol;
			++nRow;
			while (nRow != mat_src_binary_gray.rows && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow++, nCol) = 255;//用完后设置为白色
			}
			//此时遇到无效数据，往上退一格，并往左走
			--nRow;
			--nCol;
			while (nCol >= 0 && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow, nCol--) = 255;//用完后设置为白色
			}
			//此时遇到无效数据，往右退一格，往上走
			++nCol;
			--nRow;
			while (nRow >= 0 && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow--, nCol) = 255;//用完后设置为白色
			}
			//此时遇到无效数据，往下退一格，往右走
			++nRow;
			++nCol;
		}
	}

	int get_nc_effective_data(const vector<vector<Point> >& vec_contours, const Point& point_00, size_t& nLastN
		, float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data)
	{
		bool bIs_first_black = false; //第一个黑点为第一行有效数字，即N9
		bool bIs_lifting_knife = true; //遍历某列时,雕刻刀的状态，如果雕刻刀是抬刀状态，则在雕刻某点事，则在此点应该先抬刀再雕刻。
		for (int i = 0; i != vec_contours.size(); ++i)
		{
			Point2f point_dst_to_nc;
			//遍历某个连通区域的点
			for (int j = 0; j != vec_contours[i].size(); ++j)
			{
				//转换为以00点为原点的相对坐标
				point_dst_to_nc.x = (vec_contours[i][j].x - point_00.x) / fScale_pic_devide_mm;
				point_dst_to_nc.y = (point_00.y - vec_contours[i][j].y) / fScale_pic_devide_mm;

				if (false == bIs_first_black)
				{//当前黑点为全局第一个遇到的黑点
					bIs_first_black = true;
					//根据图片中的点坐标换算为G代码中的坐标，单位mm
					string str_N9 = string("N") + boost::lexical_cast<string>(nLastN++) + " G0X" + boost::lexical_cast<string>(point_dst_to_nc.x)
						+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
					str_nc_data += str_N9 + "\n";
					//缓慢下刀雕刻
					string str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
						+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + "F100";
					str_nc_data += str_row + "\n";
					bIs_lifting_knife = false;
				}
				else
				{//当前黑点为非全局第一个遇到的黑点
					//雕刻当前黑点
					string str_row;
					//判定之前刀是否为抬刀状态
					if (bIs_lifting_knife)
					{//之前处于抬刀状态，则在此点先抬刀
						str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
						str_nc_data += str_row + "\n";
						//控制下刀速度
						str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + "F100";
						str_nc_data += str_row + "\n";
					}
					str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
						+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down);
					str_nc_data += str_row + "\n";
					bIs_lifting_knife = false; //设置刀为非抬刀状态
				}
			}
			//遍历一个连通区域后，则抬刀
			//设置行以使得刀抬起
			string str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
				+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
			str_nc_data += str_row + "\n";
			bIs_lifting_knife = true;
		}
		return 0;
	}

	CBusin_OpenCV_Common_Tool()
	{
	}
	CBusin_OpenCV_Common_Tool(const CBusin_OpenCV_Common_Tool&);
	CBusin_OpenCV_Common_Tool& operator=(const CBusin_OpenCV_Common_Tool&);
private:
};
//函数原型定义
