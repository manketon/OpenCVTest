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
	static CBusin_OpenCV_Common_Tool& instance()
	{
		static CBusin_OpenCV_Common_Tool obj;
		return obj;
	}
	int get_my_RotatedRect(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3, RotatedRect& rRect);
	int draw_lines(const Mat& mat_src, const Point2f* pArr, size_t n, const Scalar& color, const string& str_win_name)
	{
		Mat mat_tmp = mat_src.clone();
		for (int i = 0; i != n; ++i)
		{
			line(mat_tmp, pArr[i],pArr[(i+1)%n], color, 2);
		}
		imshow(str_win_name, mat_tmp);
		return 0;
	}
	int test_get_binary_gray_mat(const string& str_images_dir)
	{
		//��ȡָ��Ŀ¼�е�����ͼƬ
		vector<string> vec_files_path;
		sp_boost::get_files_path_list(str_images_dir, vec_files_path, ".jpg");
		for (int i = 0; i != vec_files_path.size(); ++i)
		{
			//����ͼƬ
			Mat mat_src = imread(vec_files_path[i]);
			if (mat_src.empty())
			{
				printf("%s | read file:%s failed.", __FUNCTION__, vec_files_path[i].c_str());
				continue;
			}
			int nSize_blur = 7;
			blur(mat_src, mat_src, Size(nSize_blur, nSize_blur));
			imshow("src_mat", mat_src);
			//��ԭ�����ȡ���Ӧ�Ķ�ֵ�ҶȾ���
			Mat mat_src_gray;
			cv::cvtColor(mat_src, mat_src_gray, CV_BGR2GRAY);
			Mat mat_dst_binary;
			int nThreshold_binary = 110;
			cv::threshold(mat_src_gray, mat_dst_binary, nThreshold_binary, 255, THRESH_BINARY);
			cv::imshow("mat_dst", mat_dst_binary);
			cv::waitKey(0);
			//�����д���ļ�
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
		namedWindow("ԭʼͼ", WINDOW_NORMAL);
		imshow("ԭʼͼ", img);
		Mat out;
		//��ȡ�Զ����
		Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
		//���Ͳ���
		dilate(img, out, element);
		namedWindow("���Ͳ���", WINDOW_NORMAL);
		imshow("���Ͳ���", out);
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
		//������������ҪsrcΪ��ɫͼ
		//		scalePartAverage(mat_src, mat_dst, nResize_width * 1.0 / mat_src.cols, nResize_height * 1.0 / mat_src.rows);
		//		scaleIntervalSampling(mat_src, mat_dst, nResize_width * 1.0 / mat_src.cols, nResize_height * 1.0 / mat_src.rows);
		shrink_by_part_average(mat_src, mat_dst, nResize_width * 1.0 / mat_src.cols, nResize_height * 1.0 / mat_src.rows);
		imshow("dst", mat_dst);
		waitKey(0);
		return 0;
	}
	//����ͼ��
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
		//ֻ����uchar�͵�����
		CV_Assert(src.depth() == CV_8U);

		// ������С��ͼ��Ĵ�С
		//û���������룬��ֹ��ԭͼ�����ʱԽ��ͼ��߽�
		int rows = static_cast<int>(src.rows * yRatio);
		int cols = static_cast<int>(src.cols * xRatio);

		dst.create(rows, cols, src.type());

		const int channesl = src.channels();

		switch (channesl)
		{
		case 1: //��ͨ��ͼ��
			{
				uchar *p;
				const uchar *origal;

				for (int i = 0; i < rows; i++){
					p = dst.ptr<uchar>(i);
					//��������
					//+1 �� -1 ����ΪMat�е������Ǵ�0��ʼ������
					int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
					origal = src.ptr<uchar>(row);
					for (int j = 0; j < cols; j++){
						int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
						p[j] = origal[col];  //ȡ�ò�������
					}
				}
				break;
			}

		case 3://��ͨ��ͼ��
			{
				Vec3b *p;
				const Vec3b *origal;

				for (int i = 0; i < rows; i++) {
					p = dst.ptr<Vec3b>(i);
					int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
					origal = src.ptr<Vec3b>(row);
					for (int j = 0; j < cols; j++){
						int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
						p[j] = origal[col]; //ȡ�ò�������
					}
				}
				break;
			}
		}
	}
	//���ھֲ���ֵ��ͼ����С
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

				lastCol = col + 1; //��һ���ӿ����Ͻǵ������꣬�����겻��
			}
			lastCol = 0; //�ӿ�����Ͻ������꣬��0��ʼ
			lastRow = row + 1; //�ӿ�����Ͻ�������
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
		//ֻ����uchar�͵�����
		CV_Assert(src.depth() == CV_8U);
		int rows = static_cast<int>(src.rows * yRatio);
		int cols = static_cast<int>(src.cols * xRatio);

		dst.create(rows, cols, src.type());
		int lastRow = 0;
		int lastCol = 0;
		if (src.channels() == 1)
		{//��ͨ��
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
					lastCol = col + 1; //��һ���ӿ����Ͻǵ������꣬�����겻��
				}
				lastCol = 0; //�ӿ�����Ͻ������꣬��0��ʼ
				lastRow = row + 1; //�ӿ�����Ͻ�������
			}
		}
		else if (src.channels() == 3)
		{//��ͨ��
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
					lastCol = col + 1; //��һ���ӿ����Ͻǵ������꣬�����겻��
				}
				lastCol = 0; //�ӿ�����Ͻ������꣬��0��ʼ
				lastRow = row + 1; //�ӿ�����Ͻ�������
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
		//��������֮��
		int nSum = 0;
		for (int i = a.y; i <= b.y; ++i)
		{
			//��ָ��
			const uchar* pix = img.ptr<uchar>(i);
			for (int j = a.x; j <= b.x; ++j)
			{
				nSum += pix[j];
			}
		}
		//��Ԫ�ظ���
		int nCount = (b.x - a.x + 1) * (b.y - a.y + 1);
		nAverage = nSum / nCount;
	}
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
		//��before.jpg�м�⵽�ĵ�A��B��C��00������ ��Բ�ģ�
		cv::Point point_A_before, point_B_before, point_C_before, point_00_before;
		point_A_before = Point(1411, 645);
		point_B_before = Point(1655, 1226);
		point_C_before = Point(1343, 1188);
		point_00_before = Point(1843, 257);
		//��after.jpg�м�⵽�ĵ�A��B��C��00������ ��Բ�ģ�
		cv::Point point_A_after, point_B_after, point_C_after, point_00_after;
		point_A_after = Point(1612, 776);
		point_B_after = Point(1091, 1131);
		point_C_after = Point(1067, 816);
		point_00_after = Point(1843, 258);
		//������ֽƬ��б��AB�ĳ��ȣ���λmm
		float fAB_len = 38.25; //TODO
		//�����ֵͨ�����洫�롣
		float fScale_pic_devide_mm_before = get_euclidean(point_A_before, point_B_before) / fAB_len;
		float fScale_pic_devide_mm_after = get_euclidean(point_A_after, point_B_after) / fAB_len;
		std::cout << __FUNCTION__ << " | scale of before:" << fScale_pic_devide_mm_before << std::endl;
		std::cout << __FUNCTION__ << " | scale of after:" << fScale_pic_devide_mm_after << std::endl;
		//��ӡ��������Ϣ
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
		//��ȡ��ת�任����
		cv::Point2f points_arr_before_Tri[3] = {point_A_before, point_B_before, point_C_before}; //
		cv::Point2f points_arr_after_Trie[3] = {point_A_after, point_B_after, point_C_after};
		cv::Mat warp_mat = cv::getAffineTransform(points_arr_before_Tri, points_arr_after_Trie);
		std::cout << __FUNCTION__ << ", warp_mat" << warp_mat << std::endl;
		//����NC�����е�ĳ��
		float fLast_X =-24.683/*-26.4255*/, fLast_Y = -22.824/*-23.6664*/; //TODO
		//��fLast_X, fLast_Y���д���
		Point2f point_dst_to_nc;
		int ret = deal_nc_point(point_00_before, point_00_after, Point2f(fLast_X, fLast_Y), warp_mat
			, fScale_pic_devide_mm_before, fScale_pic_devide_mm_after, point_dst_to_nc, str_kernel_err_reason);
		//��ת�����ý������������
		float fT_X = point_dst_to_nc.x; //ת�����ֵ
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
		//nc�����еĵ���00�㣬�õ�����ԭͼ�еľ�������
		Point2f point_src_in_img; //����ԭͼ�еľ�������
		point_src_in_img.x = point_00_before.x + point_src_from_nc.x * fScale_pic_devide_mm_before;
		point_src_in_img.y = point_00_before.y - point_src_from_nc.y * fScale_pic_devide_mm_before;
		point_dst_to_nc = get_dst_point_after_affine(point_src_in_img, warp_mat);
		//ת��Ϊ��00��Ϊԭ����������
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
		//����֮�ʹ��ڵ�����
		if (dDisstance_BC + dDistance_AC <= dDistance_AB)
		{
			printf("func:%s | AC + BC <= AB.\n", __FUNCTION__);
			return false;
		}

		double dRatio_AC_AB = dDistance_AC / dDistance_AB;
		double dRatio_BC_AB = dDisstance_BC / dDistance_AB;
		//�нǶ���
		double dDegree_BAC =  acos(dDistance_AC / dDistance_AB) * 180 /CV_PI;
		double dDegree_ABC = acos(dDisstance_BC / dDistance_AB) * 180 /CV_PI;
#ifdef _DEBUG
		std::cout << "func:" << __FUNCTION__ << ", point A:" << point_A << ", point B:" << point_B << ", point C:" << point_C << std::endl;
		std::cout << "\t" <<"dDistance_AB:" << dDistance_AB << ", dDisstance_BC:" << dDisstance_BC << ", dDistance_AC:" << dDistance_AC << std::endl;
		std::cout << "\t" << "dRatio_AC_AB:" << dRatio_AC_AB << ",dRatio_BC_AB:" << dRatio_BC_AB << std::endl;
		std::cout << "\t" << "degree of BAC:" << dDegree_BAC << ", degree of ABC:" << dDegree_ABC << std::endl;
#endif
		if (fabs(dDegree_BAC - 30/*36.87*/) < dBias_degree && fabs(dDegree_ABC - 60/*53.13*/) < dBias_degree)
		{//�Ƕ������ϵ
			return true;
		}

		printf("func:%s | AC/AB:%f, BC/AB:%f\n", __FUNCTION__, dRatio_AC_AB, dRatio_BC_AB);
		return false;
	}
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
		/// ������ʾ����
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
			/// װ��ͼ��
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
		/// װ��ͼ��
		const Mat mat_src = imread(/*"wenzi.jpg"*/"./wenzi_zhongguo.jpg");
		if( !mat_src.data )
		{ 
			return -1; 
		}
		//ȥ������
		int nSize_blur = 7;
		blur(mat_src, mat_src, Size(nSize_blur, nSize_blur));
		//threshold(image,image,0,255,CV_THRESH_OTSU);
		Mat mat_dst_binary_gray;
		int nThreshold_binary = 110; //120 //��ֵԽ��˵��Խ�����ױ��ж�Ϊ��ɫ��Խ���ױ��ж�Ϊ��ɫ
		int ret = get_binary_gray_mat(mat_src, nThreshold_binary, mat_dst_binary_gray);
		if (ret != 0)
		{
			std::cout << __FUNCTION__ << " | get binary mat failed, ret:" << ret << endl; 
			return ret;
		}
		//�����õĶ�ֵ��ֵ�еĺڵ���й���һ�£�ȥ�����еĺڵ�
		//		remove_error_black_points(mat_dst_binary);
		imwrite("./wenzi_result2.jpg", mat_dst_binary_gray);
		//		std::cout << __FUNCTION__ << " | mat_dst_binary:" << mat_dst_binary << endl;
		std::cout << __FUNCTION__ << " | CV_8UC1:" << CV_8UC1 << endl;
		std::cout << __FUNCTION__ << " | mat_dst_binary type:" << mat_dst_binary_gray.type() << endl;
		std::cout << __FUNCTION__ << " | mat_dst_binary channel:" << mat_dst_binary_gray.channels() << endl;
		//����ֵ����ָ����С��������
		float fScale_pic_devide_mm = 0;
		float fMax_width_pxl = 0;
		float fMax_height_pxl = 0;
		Point point_00;
#if 0 //��
		fScale_pic_devide_mm = 244.10 / 6.73;
		fMax_width_pxl = 8.5 * fScale_pic_devide_mm;
		fMax_height_pxl = 9.0 * fScale_pic_devide_mm;
		point_00 = Point(4258, 232);
#else //�й�
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
		//�Խ���ٴν��ж�ֵ��
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
	* Brief:  ��ȡԭʼ�����Ӧ�Ķ�ֵ�ҶȾ����ȻҶȻ��ٶ�ֵ����
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const cv::Mat & mat_src -[in] ԭʼBGRͼƬ 
	*Parameter: int nThreshold_binary -[in]  ��ֵ����ֵ�����ڴ���ֵ���Ϊ255�������Ϊ0��
	*Parameter: Mat & mat_dst_binary -[out] ��ֵ�ҶȾ��� ����ɫ��255����ɫ��0�� 
	************************************/
	int get_binary_gray_mat(const cv::Mat& mat_src, int nThreshold_binary, Mat& mat_dst_binary)
	{
		Mat mat_src_gray;
		// 		/// ԭͼ��ת��Ϊ�Ҷ�ͼ��
		cvtColor( mat_src, mat_src_gray, CV_BGR2GRAY );
		//ʹ����ֵ��:������ֵ����Ϊ��ɫ��255��������Ϊ��ɫ��0��
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
		bool bIs_first_black = false; //��һ���ڵ�Ϊ��һ����Ч���֣���N9
		bool bIs_lifting_knife = true; //����ĳ��ʱ,��̵���״̬�������̵���̧��״̬�����ڵ��ĳ���£����ڴ˵�Ӧ����̧���ٵ�̡�
		string str_slowdown_flg = "F100";
		for (int i = 0; i != mat_binary.rows; ++i)
		{
			for (int j = 0; j != mat_binary.cols; ++j)
			{
				//�ж���ǰ���Ƿ�Ϊ��ɫ
				if (mat_binary.at<uchar>(i, j) - 0 < 10)
				{//��ǰ��ܿ���Ϊ��ɫ
					Point2f point_dst_to_nc;
					//ת��Ϊ��00��Ϊԭ����������,��λmm
					point_dst_to_nc.x = (j - point_00.x) / fScale_pic_devide_mm;
					point_dst_to_nc.y = (point_00.y - i) / fScale_pic_devide_mm;
					if (false == bIs_first_black)
					{//��ǰ�ڵ�Ϊȫ�ֵ�һ�������ĺڵ�
						bIs_first_black = true;
						//̧��
						string str_N9 = string("N") + boost::lexical_cast<string>(nLastN++) + " G0X" + boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
						str_nc_data += str_N9 + "\n";
						//�����µ����
						string str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + str_slowdown_flg;
						str_nc_data += str_row + "\n";
						bIs_lifting_knife = false;
					}
					else
					{//��ǰ�ڵ��ȫ�ֵ�һ�������ĺڵ�
						//��̵�ǰ�ڵ�
						string str_row;
						//�ж�֮ǰ���Ƿ�Ϊ̧��״̬
						if (bIs_lifting_knife)
						{//֮ǰ����̧��״̬�����ڴ˵���̧��
							str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
								+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
							str_nc_data += str_row + "\n";
							//�ٽ��ٵ��
							str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
								+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + str_slowdown_flg;
							str_nc_data += str_row + "\n";
						}
						str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down);
						str_nc_data += str_row + "\n";
						bIs_lifting_knife = false; //���õ�Ϊ��̧��״̬
					}
					//�ж���һ������ڣ����ж����Ƿ�Ϊ��ɫ�����Ϊ��ɫ������Ҫ���̧������
					if (j + 1 != mat_binary.cols && 255 - mat_binary.at<uchar>(i, j + 1) < 10)
					{//��һ����δԽ����Ϊ��ɫ,����Ҫ̧��
						//��������ʹ�õ�̧��
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
	* Brief:  ��ȡ�ļ�ͷ����һ��λ��
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
			+ "N7 ([׶��ƽ��]JD-45-0.10)\n"
			+ "N8 G17\n";
		nLastN = 9; //�ļ�ͷ�ܺ���+1
		return str_head_flg;
	}
	/************************************
	* Method:    get_nc_tail
	* Brief:  ���ݵ�ǰ���±�������NC�����β��
	* Access:    protected 
	* Returns:   std::string
	* Qualifier:
	*Parameter: const size_t & nLastN -[in] ��ǰ������+1  
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
		//������ֵ���󣬴�ʱҪ�����ִ�Ϊ��ɫ��������Ϊ��ɫ��
		vector<Point> vec_writting_points; //���ֶ�Ӧ�ĵ��б�
		int  nCircle_radius = 8; //��ֵҪ���ڻ�ɫԲ�뾶����λmm
		int n00_threshed = nCircle_radius * fScale_pic_devide_mm;
		for (size_t nRow = 0; nRow != mat_src_binary_gray.rows; ++nRow)
		{
			for (size_t nCol = 0; nCol != mat_src_binary_gray.cols; ++nCol)
			{
				//����00�㵽���Ͻ��������,���⽫00��Ҳ����Ϊ������
				if (nCol > point_00.x - n00_threshed && nRow < point_00.y + n00_threshed)
				{
					continue;
				}
				//�ж��Ƿ�Ϊ��ɫ��
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
		//��ʾԭͼ
		//		imshow("mat_src_binary", mat_src_binary);
		//���ҵ㼯����С��Ӿ���
		RotatedRect rect_points = minAreaRect(vec_writting_points);
		//����һ���洢�����ĸ��������ı���
		Point2f points_rect_list[4]; //The order is bottomLeft, topLeft, topRight, bottomRight
		ret = get_rect_points(mat_src_binary_gray, rect_points, points_rect_list, 4);
		if (ret != 0)
		{
			std::cout << __FUNCTION__ << " | error, line:" << __LINE__ << endl;
			return ret;
		}
		//��������
		draw_lines(mat_src_binary_gray, points_rect_list, 4, "./img_with_line.jpg");
		//��ȡ��С���ű���
		float fHeight_pxl = get_euclidean((Point)(points_rect_list[0]), (Point)(points_rect_list[1]));
		float fWidth_pxl = get_euclidean((Point)(points_rect_list[1]), (Point)(points_rect_list[2]));
		double fx = (fMax_width_pxl - 0.5 * fScale_pic_devide_mm) / fWidth_pxl; //�����߽��Ԥ��һ������
		double fy = (fMax_height_pxl - 0.5 * fScale_pic_devide_mm) / fHeight_pxl;
		double fScale = std::min(fx, fy); //����Сֵ��Ϊ��������ű�����ע�⣺x��y���򲻿�ʹ�ò�ͬ�����ű������������
		//��������ת��ˮƽ����������
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
		//��������
		//Ѱ�����������
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		Mat mat_sub(mat_src_binary(Rect(0, 0, point_00.x - 200, mat_src_binary.rows)));
		findContours(mat_sub, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_NONE, Point());
		Mat imageContours=Mat(mat_sub.size(),CV_8UC1, Scalar(255));
		//������С��Ӿ���
		Point2f P[4];
		for (int i = 0; i != contours.size(); ++i)
		{
			//��������
			//��������
			drawContours(imageContours, contours, i, Scalar(0), 1, 8, hierarchy);
			//������������С������
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
		//��������ɫ��������
		cv::Mat image = cv::Mat::zeros(512, 512, CV_8UC3);
		image.setTo(cv::Scalar(100, 0, 0));
		cv::imshow("original", image);

		//��ȡ������ͼƬ
		cv::Mat roi = cv::imread("E:\\Images\\Hepburn.png", cv::IMREAD_COLOR);
		cv::imshow("roi", roi);

		//���û����������򲢸���
		cv::Rect roi_rect = cv::Rect(128, 128, roi.cols, roi.rows);
		roi.copyTo(image(roi_rect));

		cv::imshow("result", image);

		cv::waitKey(0);

		return 0;
	}
	int remove_error_black_points(Mat& mat_binary)
	{
		size_t nStep = 5;
		//����Mat�е�ÿһ���㣬��������Χ�ĺڵ���Ŀ�����������ֵ������Ϊ���������һ����
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
	* Brief:  ������mat�У��Դ˵�Ϊ���ģ�x��y����Ĳ���nStep������Χ�ڣ��ڵ�ĸ���
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
				{//�Ǻ�ɫ
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

	//��������ת��ˮƽ����
	int rotate_writting_to_horizontal_direction(const Mat& src_mat, const RotatedRect& rRect
		, const float& fScale, const float& fScale_pic_devide_mm, Mat& mat_dst)
	{
		//��ȡ�任ǰ����С��Ӿ��ε��ĸ�����
		Point2f src_rect_points_arr[4]; //The order is bottomLeft, topLeft, topRight, bottomRight
		int ret = get_rect_points(src_mat, rRect, src_rect_points_arr, 4);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | error, line:" << __LINE__ << endl;
			return ret;
		}
		//����Ӿ��ε�������Ϊ��ת����
		CvPoint2D32f  center = cvPoint2D32f(rRect.center.x, rRect.center.y);
		double Ddgree = 0; //Ϊ���ı�ʾ��ʱ����ת,��λ�Ƕ�
		if (rRect.center.x < src_rect_points_arr[0].x)
		{
			Ddgree = -(90 + rRect.angle);
		}
		else
		{
			Ddgree = -rRect.angle;
		}
		Mat mat_rotated;//�任���þ���
		//Ϊ��������챳��������ɫ��������С��������Դ�����Сһ��
		mat_dst = Mat(src_mat.size(),src_mat.type(), Scalar(255));
		//��ȡ��������Լ���ת�����Ĵ�С
		Mat mat_rotation; //�������
		Size Dsize;
		//������ת���ġ����ű�����ȡ����任����ͽ������Ĵ�С
		ret = CBusin_Opencv_Transform_Tool::instance().get_rotation_matrix_without_loss(src_mat, center, Ddgree, fScale, mat_rotation, Dsize);
		if (ret)
		{
			std::cout << " | error, line:" << __LINE__ << ", ret:" << ret << endl;
			return ret;
		}
		//��ת�任���԰�����Ӿ��Σ��������־�������
		cv::warpAffine(src_mat, mat_rotated, mat_rotation, Dsize, 1, 0, 0);
		imwrite("./mat_rotated.jpg", mat_rotated);

		Point2f dst_rect_points_arr[4];
		//��ȡ�任����С��Ӿ��ζ��ڵ��ĸ�����
		ret = get_points_after_affined(mat_rotation, src_rect_points_arr, 4, dst_rect_points_arr);
		//��ʱ��Ӿ���������
		//�������б����������ʹ��˳��ΪbottomLeft, topLeft, topRight, bottomRight
		ret = sort_rect_points(dst_rect_points_arr, 4);
		//����Ӿ��ηŴ�,�����ڵ�������
		int nExt_size_pxl = 0.5 * fScale_pic_devide_mm; //�����߷Ŵ�Ĵ�С����λ����
		ret = increase_rect_size(dst_rect_points_arr, 4, nExt_size_pxl);
		//����ת�����Ӿ��λ�����
		ret = draw_lines(mat_rotated, dst_rect_points_arr, 4, "./img_with_line_after_retated.jpg");
		//�����ֶ�Ӧ��������ȡ����
		//��ȡ���������Ӧ�ľ���
		size_t nWidth_roi = dst_rect_points_arr[2].x - dst_rect_points_arr[1].x /*rRect.size.width * fScale*/; //����������
		size_t nHeight_roi = dst_rect_points_arr[3].y - dst_rect_points_arr[2].y /*rRect.size.height * fScale*/;
		Rect dst_rect_roi(dst_rect_points_arr[1].x, dst_rect_points_arr[1].y, nWidth_roi, nHeight_roi);
		//�����������
		Mat mat_roi(mat_rotated(dst_rect_roi));
		imwrite("./mat_roi.jpg", mat_roi);
		//����ȡ�����������򸲸ǵ����������У��Եõ����ս������
		//��������е�Ŀ�����򣨼������ڽ�������е�λ�ã�
		Rect src_rect_roi(rRect.center.x - rRect.size.width / 2, rRect.center.y - rRect.size.height / 2, nWidth_roi, nHeight_roi);
		mat_roi.copyTo(mat_dst(src_rect_roi));
		return 0;
	}

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
	* Brief:  ����RotatedRect��ȡ��Ӿ��ε��ĸ�����
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: const Mat & mat_src -[in] Դ���� 
	*Parameter: const RotatedRect & rRect -[in] ��С��Ӿ��� 
	*Parameter: Point2f * pts -[out] ��С��Ӿ��ζ�Ӧ�� 
	*Parameter: size_t nPoints_num -[in] ���ζ�����Ŀ 
	************************************/
	int get_rect_points(const Mat& mat_src, const RotatedRect& rRect, Point2f* pts, size_t nPoints_num = 4)
	{
		//��rect_points�����д洢������ֵ�ŵ�pts������
		rRect.points(pts);
		//У�����õĵ��Ƿ�Ϸ���������ڷǷ�ֵ���򱨴�
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
	* Brief:  �Ծ��εĶ����б����������ʹ�����˳��ΪbottomLeft, topLeft, topRight, bottomRight
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  ��Ϊ����ʱ�����εĶ������飻��Ϊ���ʱ���ĸ����㱻������
	*Parameter: size_t nPoints_num -[in] �������
	************************************/
	int sort_rect_points(Point2f* pRect_points, size_t nPoints_num)
	{
		Point2f bottomLeft, topLeft, topRight, bottomRight;
		bottomLeft= topLeft	= topRight = bottomRight = pRect_points[0];
		float fThreshold = 0.01; //����������ж�ʱ����ֵ
		for (size_t i = 0; i != nPoints_num; ++i)
		{
			if ((pRect_points[i].x < bottomLeft.x || fabs(pRect_points[i].x - bottomLeft.x) < fThreshold)
				&& (pRect_points[i].y > bottomLeft.y || fabs(pRect_points[i].y - bottomLeft.y) < fThreshold))
			{//bottomLeft��x��С��y���
				bottomLeft = pRect_points[i];
			}
			if ((pRect_points[i].x < topLeft.x || fabs(pRect_points[i].x - topLeft.x) < fThreshold) 
				&& (pRect_points[i].y < topLeft.y || fabs(pRect_points[i].y - topLeft.y) < fThreshold))
			{//topLeft��x��С��y��С
				topLeft = pRect_points[i];
			}
			if ((pRect_points[i].x > topRight.x || fabs(pRect_points[i].x - topRight.x) <fThreshold) 
				&& (pRect_points[i].y < topRight.y || fabs(pRect_points[i].y - topRight.y ) < fThreshold))
			{//topRight��x�����y��С
				topRight = pRect_points[i];
			}
			if ((pRect_points[i].x > bottomRight.x || fabs(pRect_points[i].x - bottomRight.x) < fThreshold)
				&& (pRect_points[i].y > bottomRight.y || fabs(pRect_points[i].y - bottomRight.y) < fThreshold))
			{//bottomRight��x�����y���
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
	* Brief:  �����ε��ĸ����㣨˳��ΪbottomLeft, topLeft, topRight, bottomRight���ֱ���x��y������ƽ��nExt_size_pxl�����أ���ʹ���������η�Χ���
	* Access:    protected 
	* Returns:   int
	* Qualifier:
	*Parameter: Point2f * pRect_points -[in/out]  ���ζ���
	*Parameter: size_t nPoints_num -[in/out]  
	*Parameter: int nExt_size_pxl -[in/out]  
	************************************/
	int increase_rect_size(Point2f* pRect_points, size_t nPoints_num, int nExt_size_pxl)
	{
		//bottomLeft��x��С��y���
		pRect_points[0].x -= nExt_size_pxl;
		pRect_points[0].y += nExt_size_pxl;
		//topLeft��x��С��y��С
		pRect_points[1].x -= nExt_size_pxl;
		pRect_points[1].y -= nExt_size_pxl;
		//topRight��x�����y��С
		pRect_points[2].x += nExt_size_pxl;
		pRect_points[2].y -= nExt_size_pxl;
		//bottomRight��x�����y���
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
				{//��ǰ��Ϊ��ɫ
					vector<Point> vec_points;
					traverse_connected_graph(mat_tmp_binary_gray, Point(nCol, nRow), vec_points);
					vec_contours.push_back(vec_points);
				}
			}
		}
	}
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
	void traverse_connected_graph(Mat& mat_src_binary_gray, const Point& point_start, vector<Point>& vec_points)
	{
		int nRow = point_start.y, nCol = point_start.x;
		while (nRow != mat_src_binary_gray.rows  && nRow >= 0 && nCol >= 0 && nCol != mat_src_binary_gray.cols 
			&& mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
		{
			//һֱ������
			while (nCol != mat_src_binary_gray.cols && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow, nCol++) = 255;//���������Ϊ��ɫ
			}
			//��ʱ������Ч���ݣ�������һ�񣬲�������
			--nCol;
			++nRow;
			while (nRow != mat_src_binary_gray.rows && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow++, nCol) = 255;//���������Ϊ��ɫ
			}
			//��ʱ������Ч���ݣ�������һ�񣬲�������
			--nRow;
			--nCol;
			while (nCol >= 0 && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow, nCol--) = 255;//���������Ϊ��ɫ
			}
			//��ʱ������Ч���ݣ�������һ��������
			++nCol;
			--nRow;
			while (nRow >= 0 && mat_src_binary_gray.at<uchar>(nRow, nCol) < 10)
			{
				vec_points.push_back(Point(nCol, nRow));
				mat_src_binary_gray.at<uchar>(nRow--, nCol) = 255;//���������Ϊ��ɫ
			}
			//��ʱ������Ч���ݣ�������һ��������
			++nRow;
			++nCol;
		}
	}

	int get_nc_effective_data(const vector<vector<Point> >& vec_contours, const Point& point_00, size_t& nLastN
		, float fScale_pic_devide_mm, float fZ_up, float fZ_down, string& str_nc_data)
	{
		bool bIs_first_black = false; //��һ���ڵ�Ϊ��һ����Ч���֣���N9
		bool bIs_lifting_knife = true; //����ĳ��ʱ,��̵���״̬�������̵���̧��״̬�����ڵ��ĳ���£����ڴ˵�Ӧ����̧���ٵ�̡�
		for (int i = 0; i != vec_contours.size(); ++i)
		{
			Point2f point_dst_to_nc;
			//����ĳ����ͨ����ĵ�
			for (int j = 0; j != vec_contours[i].size(); ++j)
			{
				//ת��Ϊ��00��Ϊԭ����������
				point_dst_to_nc.x = (vec_contours[i][j].x - point_00.x) / fScale_pic_devide_mm;
				point_dst_to_nc.y = (point_00.y - vec_contours[i][j].y) / fScale_pic_devide_mm;

				if (false == bIs_first_black)
				{//��ǰ�ڵ�Ϊȫ�ֵ�һ�������ĺڵ�
					bIs_first_black = true;
					//����ͼƬ�еĵ����껻��ΪG�����е����꣬��λmm
					string str_N9 = string("N") + boost::lexical_cast<string>(nLastN++) + " G0X" + boost::lexical_cast<string>(point_dst_to_nc.x)
						+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
					str_nc_data += str_N9 + "\n";
					//�����µ����
					string str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
						+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + "F100";
					str_nc_data += str_row + "\n";
					bIs_lifting_knife = false;
				}
				else
				{//��ǰ�ڵ�Ϊ��ȫ�ֵ�һ�������ĺڵ�
					//��̵�ǰ�ڵ�
					string str_row;
					//�ж�֮ǰ���Ƿ�Ϊ̧��״̬
					if (bIs_lifting_knife)
					{//֮ǰ����̧��״̬�����ڴ˵���̧��
						str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_up);
						str_nc_data += str_row + "\n";
						//�����µ��ٶ�
						str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
							+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down) + "F100";
						str_nc_data += str_row + "\n";
					}
					str_row = string("N") + boost::lexical_cast<string>(nLastN++) + " X" +boost::lexical_cast<string>(point_dst_to_nc.x)
						+ "Y" + boost::lexical_cast<string>(point_dst_to_nc.y) + "Z" + boost::lexical_cast<string>(fZ_down);
					str_nc_data += str_row + "\n";
					bIs_lifting_knife = false; //���õ�Ϊ��̧��״̬
				}
			}
			//����һ����ͨ�������̧��
			//��������ʹ�õ�̧��
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
//����ԭ�Ͷ���
