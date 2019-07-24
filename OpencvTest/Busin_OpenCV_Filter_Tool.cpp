/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Filter_Tool.cpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2019/06/19
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/06/19	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "Busin_OpenCV_Filter_Tool.h"
#include "ImageMaster/include/ImageMaster.h"
#include <string>
#include "Boost_Common_Tool.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include "Busin_OpenCV_Common_Tool.h"
#include <boost/math/constants/constants.hpp>
#include <boost/algorithm/clamp.hpp>
#include "Resource_Manager.h"
using namespace cv;
using namespace std;

#define GAMMA           1.0
#define EPSILON         2

CBusin_OpenCV_Filter_Tool_Inst& CBusin_OpenCV_Filter_Tool_Inst::instance()
{
	return ms_inst;
}
//素描
int CBusin_OpenCV_Filter_Tool_Inst::test_sketch()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat src = imread(str_img_path,1);
		int width = src.cols;
		int heigh = src.rows;
		Mat gray0, gray1;
		//去色
		cvtColor(src, gray0, CV_BGR2GRAY);
		//反色
		addWeighted(gray0, -1, NULL, 0, 255, gray1);
		//高斯模糊,高斯核的Size与最后的效果有关
		int nKernel_size = 11;
		GaussianBlur(gray1, gray1, Size(nKernel_size, nKernel_size), 0);
		imshow("result of gauss", gray1);
		//融合：颜色减淡
		Mat img(gray1.size(), CV_8UC1);
		for (int y = 0; y < heigh; ++y)
		{

			uchar* P0  = gray0.ptr<uchar>(y);
			uchar* P1  = gray1.ptr<uchar>(y);
			uchar* P  = img.ptr<uchar>(y);
			for (int x = 0; x < width; ++x)
			{
				int tmp0 = P0[x];
				int tmp1 = P1[x];
				P[x] =(uchar) min((tmp0 + (tmp0 * tmp1)/(256 - tmp1)), 255);
			}

		}
		Mat img_for_show;
		cv::resize(img, img_for_show, Size(img.cols / 3, img.rows / 4));
		imshow("素描", img_for_show);
		//二值化
		Mat mat_binary;
		cv::threshold(img, mat_binary, 245, 255, cv::THRESH_BINARY);
		imshow("mat_binary", mat_binary);
		waitKey();
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path, mat_binary);
	}
	return 0;
}

//衣服颜色去除了，但是眼珠部分与PS不同
int CBusin_OpenCV_Filter_Tool_Inst::test_Laplacian_sketch()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat src = imread(str_img_path,1);
		int width = src.cols;
		int heigh = src.rows;
		Mat gray0, gray1;
		//去色
		cvtColor(src, gray0, CV_BGR2GRAY);
		const int MEDIAN_BLUR_FILTER_SIZE = 3;
//		cv::medianBlur(gray0, gray0, MEDIAN_BLUR_FILTER_SIZE);
		GaussianBlur(gray0, gray0, Size(MEDIAN_BLUR_FILTER_SIZE, MEDIAN_BLUR_FILTER_SIZE), 0);
		cv::Mat edges;
		//边缘检测
		const int LAPLACIAN_FILTER_SIZE = 3;
		cv::Laplacian(gray0, edges, CV_8U, LAPLACIAN_FILTER_SIZE);
//		imshow("edges", edges);
		cv::Mat mask;
		const int EDGE_THRESHOLD = 10;
		cv::threshold(edges, mask, EDGE_THRESHOLD, 255, cv::THRESH_BINARY_INV);
// 		imshow("素描",mask);
// 		waitKey();
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path, mask);
	}
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_Sobel_sketch()
{
	//Sobel
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat src = imread(str_img_path,1);
		int width = src.cols;
		int heigh = src.rows;
		Mat gray0, gray1;
		//去色
		cvtColor(src, gray0, CV_BGR2GRAY);
		const int MEDIAN_BLUR_FILTER_SIZE = 3;
		//		cv::medianBlur(gray0, gray0, MEDIAN_BLUR_FILTER_SIZE);
		GaussianBlur(gray0, gray0, Size(MEDIAN_BLUR_FILTER_SIZE, MEDIAN_BLUR_FILTER_SIZE), 0);
		cv::Mat edges;
		//边缘检测
		const int LAPLACIAN_FILTER_SIZE = 3;
		cv::Sobel(gray0, edges, CV_8U, 1, 0, LAPLACIAN_FILTER_SIZE);

		imshow("edges", edges);
		cv::Mat mask;
		const int EDGE_THRESHOLD = 10;
		cv::threshold(edges, mask, EDGE_THRESHOLD, 255, cv::THRESH_BINARY_INV);
		imshow("素描",mask);
		waitKey();
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path, mask);
	}
	return 0;
}

//效果不咋地
int CBusin_OpenCV_Filter_Tool_Inst::test_differenceOfGaussian()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output/differenceOfGaussian";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat src = imread(str_img_path,1);
		Mat mat_sketch;
		differenceOfGaussian(src, mat_sketch);
		Mat img_for_show;
		cv::resize(mat_sketch, img_for_show, Size(mat_sketch.cols / 3, mat_sketch.rows / 4));
		imshow("素描", img_for_show);
		//二值化
		Mat mat_binary;
		cv::threshold(mat_sketch, mat_binary, 7, 255, cv::THRESH_BINARY_INV);
 		imshow("mat_binary", mat_binary);
 		waitKey();
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path, mat_binary);
	}
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_difference_IPLB()
{
	std::string str_img_path = "C:/Users/dell.dell-PC/Desktop/1301_IPLB_result.jpg";
	Mat mat_src = imread(str_img_path, 1);
	Mat mat_sketch;
 	//增加光照和对比度
 	CBusin_OpenCV_Common_Tool::instance().change_contrast_and_brightness(mat_src, 5, 20, mat_sketch);
	cv::cvtColor(mat_sketch, mat_sketch, CV_BGR2GRAY);
	resize(mat_sketch, mat_sketch,Size(mat_sketch.cols / 4, mat_sketch.rows / 4));
	imshow("素描", mat_sketch);
	imwrite(str_img_path + "_change_CB.jpg", mat_sketch);
	//二值化
	Mat mat_binary;
	//高斯滤波时，阈值为50时较好。
	cv::threshold(mat_sketch, mat_binary, 55, 255, cv::THRESH_BINARY_INV);
	imshow("mat_binary after threshold", mat_binary);
 	//高斯滤波
 	cv::GaussianBlur(mat_binary, mat_binary, Size(3, 3), 1, 1);
	waitKey();
	imwrite(str_img_path + "_result.jpg", mat_binary);
	return 0;
}

//高反差保留
int CBusin_OpenCV_Filter_Tool_Inst::test_GaoFanChaBaoLiu()
{
	int R = 11;
	std::string str_img_path = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/lhj.jpg";
	Mat src = imread(str_img_path, 1);
	imshow("src",src);
	int width=src.cols;
	int heigh=src.rows;
	Mat img;
	src.copyTo(img);
	Mat avg;
	GaussianBlur(img,avg,Size(R,R),0.0);
//	blur(img, avg, Size(R, R));
	Mat dst(img.size(),CV_8UC3);
	float tmp;
	for (int y=0;y<heigh;y++)
	{
		uchar* imgP=img.ptr<uchar>(y);
		uchar* avgP=avg.ptr<uchar>(y);
		uchar* dstP=dst.ptr<uchar>(y);
		for (int x=0;x<width;x++)
		{

			float r0 = ((float)imgP[3*x]-(float)avgP[3*x]);  
			tmp = 128+abs(r0)*r0/(2*R);
			tmp=tmp>255?255:tmp;
			tmp=tmp<0?0:tmp;
			dstP[3*x]=(uchar)(tmp);

			float r1 = ((float)imgP[3*x+1]-(float)avgP[3*x+1]);
			tmp = 128+abs(r1)*r1/(2*R);
			tmp=tmp>255?255:tmp;
			tmp=tmp<0?0:tmp;
			dstP[3*x+1]=(uchar)(tmp);

			float r2 = ((float)imgP[3*x+2]-(float)avgP[3*x+2]);
			tmp = 128+abs(r2)*r2/(2*R);
			tmp=tmp>255?255:tmp;
			tmp=tmp<0?0:tmp;
			dstP[3*x+2]=(uchar)(tmp);
		}
	}
	imshow("high",dst);

	//高通滤波测试

	Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
		-1,  5, -1,
		0, -1,  0);
	Mat dstF;
	filter2D(img,dstF,img.depth(),kern);
	imshow("kernel",dstF);

	waitKey();
	//imwrite("D:/高反差保留.jpg",dst);
	//	imwrite("D:/高通滤波.jpg",dstF);
	imwrite(str_img_path + "_result.jpg", dst);
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_photocopy()
{
	Mat src = imread("C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/src.jpg", 1);
	Mat dst(src.size(), CV_8UC1);
	int Stride = src.cols * src.channels();
	int yushu = Stride % 4;
	if (yushu != 0)
	{
		Stride = Stride + 4 - yushu;
	}
//	IM_PhotoCopy(src.data, dst.data, src.cols, src.rows, Stride, 3, 8, 255, 0);
	imshow("result of photocoyp", dst);
	waitKey();
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_photocopy_myself()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output/myself_photocopy";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat mat_src_bgr = imread(str_img_path, 1);
		int width = mat_src_bgr.cols;
		int heigh = mat_src_bgr.rows;
		Mat mat_dst_gray;
		Mat mat_src_gray;
		cv::cvtColor(mat_src_bgr, mat_src_gray, CV_BGR2GRAY);
		SPhotocopy_Vals pvals =
		{
			8.0,  /* mask_radius */
			0.8,  /* sharpness */
			0.75, /* threshold */
			0.2,  /* pct_black */
			0.2   /* pct_white */
		};
		photocopy_myself(mat_src_gray, pvals, mat_dst_gray);
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path + "_sumiao.jpg", mat_dst_gray);
	}
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_photocopy_GIMP()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output/gimp_photocopy";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat mat_src_bgr = imread(str_img_path, 1);
		int width = mat_src_bgr.cols;
		int heigh = mat_src_bgr.rows;
		Mat mat_dst_gray;
		Mat mat_src_gray;
		cv::cvtColor(mat_src_bgr, mat_src_gray, CV_BGR2GRAY);
		SPhotocopy_Vals pvals =
		{
			8.0,  /* mask_radius */
			0.8,  /* sharpness */
			0.75, /* threshold */
			0.2,  /* pct_black */
			0.2   /* pct_white */
		};
		photocopy_gimp(mat_src_gray, pvals, mat_dst_gray);
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path + "_sumiao.jpg", mat_dst_gray);
	}
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_difference_Edge_Detect()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/滤镜开发图片/开发/test_output/difference_Edge_Detect";
	vector<string> vec_files_path;
	int ret = sp::get_filenames(str_test_imgs_dir, vec_files_path);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | error, ret:" << ret << endl;
		return ret;
	}
	for (int i = 0; i != vec_files_path.size(); ++i)
	{
		std::string str_img_path = vec_files_path[i];
		Mat mat_src_bgr = imread(str_img_path, 1);
		int width = mat_src_bgr.cols;
		int heigh = mat_src_bgr.rows;
		//TODO::增强对比度和亮度
		Mat mat_src_gray;
		cv::cvtColor(mat_src_bgr, mat_src_gray, CV_BGR2GRAY);
		Mat mat_dst_gray;
		difference_Edge_Detect(mat_src_gray, mat_dst_gray, Rect(0, 0, width, heigh));
		Mat img_for_show;
		cv::resize(mat_dst_gray, img_for_show, Size(mat_dst_gray.cols / 3, mat_dst_gray.rows / 4));
		imshow("素描", img_for_show);
		//二值化
		Mat mat_binary;
		//TODO::阈值应该和原图有关，如何设置这个阈值呢？
		int nThreshold = 20; //值越大，细节越少
		cv::threshold(mat_dst_gray, mat_binary, nThreshold, 255, cv::THRESH_BINARY_INV);
// 		imshow("mat_binary", mat_binary);
// 		waitKey();
		string str_dst_img_path = str_output_dir + "/" + boost::filesystem::path(str_img_path).filename().string()
			+  "_result.jpg";
		imwrite(str_dst_img_path, mat_binary);
		Mat mat_inverted_gray;//反色的灰度图
		//反色
		addWeighted(mat_dst_gray, -1, NULL, 0, 255, mat_inverted_gray);
		imwrite(str_dst_img_path + "_sumiao.jpg", mat_inverted_gray);
	}
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_draw_chin()
{
	//读取人物图
	Mat mat_src_gray = imread("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_src/0620151349.jpg", IMREAD_GRAYSCALE);
	if (mat_src_gray.empty())
	{
		std::cout <<__FUNCTION__ << "| fail to read image" << endl;
		return -1;
	}

	//下巴顶点列表
	vector<cv::Point2f> vec_chin_points;
	vec_chin_points.push_back(Point2f(1107.26, 1373.39));
	vec_chin_points.push_back(Point2f(1118.8, 1422.04));
	vec_chin_points.push_back(Point2f(1131, 1470.01));
	vec_chin_points.push_back(Point2f(1145.9, 1518.16));
	vec_chin_points.push_back(Point2f(1161.97, 1565.18));
	vec_chin_points.push_back(Point2f(1179.12, 1613.65));
	vec_chin_points.push_back(Point2f(1200.39, 1658.55));
	vec_chin_points.push_back(Point2f(1234.04, 1706.42));
	vec_chin_points.push_back(Point2f(1275.61, 1744.79));
	vec_chin_points.push_back(Point2f(1323.92, 1778.03));
 	vec_chin_points.push_back(Point2f(1378.02, 1797.53));
	//chin
	vec_chin_points.push_back(Point2f(1428.18, 1807.07));
	vec_chin_points.push_back(Point2f(1480.7,  1801.78));
	vec_chin_points.push_back(Point2f(1532.58, 1792.03));
	//left cheek
	vec_chin_points.push_back(Point2f(1577.94, 1769.14));
	vec_chin_points.push_back(Point2f(1624.8, 1735.83));
	vec_chin_points.push_back(Point2f(1663.42, 1693.04));
	vec_chin_points.push_back(Point2f(1694.84, 1644.08));
	vec_chin_points.push_back(Point2f(1714.44, 1589.13));
	vec_chin_points.push_back(Point2f(1722.26, 1538.31));
	vec_chin_points.push_back(Point2f(1724.85, 1486.92));
	vec_chin_points.push_back(Point2f( 1726.79, 1435.85));
	vec_chin_points.push_back(Point2f( 1727.85, 1386.39));
	vec_chin_points.push_back(Point2f(1725.72, 1335.47));
	vec_chin_points.push_back(Point2f(1723.16, 1286.86));
	//画出人脸检测结果点
	for (int i = 0; i != vec_chin_points.size(); ++i)
	{
		mat_src_gray.at<uchar>(vec_chin_points[i]) = 255;
		//为了使得特征点更加突出
		mat_src_gray.at<uchar>(vec_chin_points[i] + Point2f(0, 1)) = 255;
		mat_src_gray.at<uchar>(vec_chin_points[i] - Point2f(0, 1)) = 255;
		mat_src_gray.at<uchar>(vec_chin_points[i] - Point2f(1, 0)) = 255;
		mat_src_gray.at<uchar>(vec_chin_points[i] + Point2f(1, 0)) = 255;
		mat_src_gray.at<uchar>(vec_chin_points[i] - Point2f(1, 1)) = 255;
		mat_src_gray.at<uchar>(vec_chin_points[i] + Point2f(1, 1)) = 255;
	}
	//使用CatmullRom插样算法来获取下巴顶点之间曲线上的点
	vector<Point2f> vec_cure_points;
	for (int i = 0; i != vec_chin_points.size() - 3; ++i)
	{
		Point2f p0 = vec_chin_points[i];
		Point2f p1 = vec_chin_points[i + 1];
		Point2f p2 = vec_chin_points[i + 2];
		Point2f p3 = vec_chin_points[i + 3];
		CBusin_OpenCV_Common_Tool::instance().get_CatmullRom_points(p0, p1, p2, p3, 500, vec_cure_points);
	}
	//画曲线
	for (int i = 0; i != vec_cure_points.size(); ++i)
	{
		mat_src_gray.at<uchar>(vec_cure_points[i]) = 255;
	}
	cv::imwrite("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_result/chin.jpg", mat_src_gray);
	return 0;
}

void CBusin_OpenCV_Filter_Tool_Inst::Sketch(const Mat& img, Mat& dst)
{
	if ( dst.empty())
		dst.create(img.rows, img.cols, img.type());	

	int height = img.rows;
	int width = img.cols;
	int chns = img.channels();
	int border = 1;
	int i, j, k;

	for ( i=border; i<height-border; i++)
	{
		unsigned char* dstData = (unsigned char*)dst.data + dst.step*i;
		for ( j=border; j<width-border; j++)
		{				
			for ( k=0; k<chns; k++)
			{
				int sum = 8*getPixel(img, i, j, k) - getPixel(img, i-1, j-1, k) - getPixel(img, i-1, j, k) - getPixel(img, i-1, j+1, k) 
					- getPixel(img, i, j-1, k) - getPixel(img, i, j+1, k)
					- getPixel(img, i+1, j-1, k) - getPixel(img, i+1, j, k) - getPixel(img, i+1, j+1, k) ;

				//dstData[j*chns+k] = saturate_cast<uchar>(dstData[j*chns+k] + sum);				
				dstData[j*chns+k] = saturate_cast<uchar>(sum) ;
			}				
		}
	}
}

void CBusin_OpenCV_Filter_Tool_Inst::DiamondEmboss(const Mat& img, Mat& dst, EMBOSS_DIRECTION Ed /* = SE */,int offset /* = 127 */)
{
	//https://blog.csdn.net/kezunhai/article/details/41786571
	if ( dst.empty())
		dst.create(img.rows, img.cols, img.type());	

	int height = img.rows;
	int width = img.cols;
	int chns = img.channels();
	int border = 1;
	int i, j, k, sum;

	int ioffset = 0; // 根据方位来偏移
	int joffset = 0;  

	switch(Ed)
	{
	case  N:
		ioffset = -1;
		joffset = 0;
		break;
	case NE:
		ioffset = -1;
		joffset = 1;
		break;
	case E:
		ioffset = 0;
		joffset = 1;
		break;
	case SE:
		ioffset = 1;
		joffset = 1;
		break;
	case S:
		ioffset = 1;
		joffset = 0;
		break;
	case SW:
		ioffset = 1;
		joffset = -1;
		break;
	case W:
		ioffset = 0;
		joffset = -1;
		break;
	case NW:
		ioffset = -1;
		joffset = -1;
	default:
		ioffset = 1;
		joffset = 1;
		break;
	}

	for (  i= border; i<height-border; ++i)
	{		
		unsigned char* dstData = (unsigned char*)dst.data+dst.step*i;
		for ( j=border; j<width-border; ++j)
		{			
			for ( k=0; k<chns; k++)
			{
				sum = getPixel(img, i, j, k) - getPixel(img, i-ioffset, j-joffset, k) +offset;

				dstData[j*chns+k] = saturate_cast<uchar>(sum);	

			}		
		}
	}
}

int CBusin_OpenCV_Filter_Tool_Inst::getPixel(const Mat& mat_img, int y, int x, int channel)
{
	if (mat_img.channels() == 3 )
	{
		return mat_img.at<Vec3b>(y, x)[channel];
	}
	else if (mat_img.type() ==  CV_8UC1) 
	{
		return mat_img.at<uchar>(y, x);
	}
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::differenceOfGaussian(const Mat& mat_src, Mat& mat_dst)
{
	//将原图转换为灰度图
	Mat mat_gray;
	cv::cvtColor(mat_src, mat_gray, COLOR_BGR2GRAY);
	//用两个不同的模糊半径对灰度图像执行高斯模糊（取得两幅高斯模糊图像）
	//TODO::不同的半径以及sigma对结果都有影响
	Mat mat_gaussian_blur_1;
	int GAUSSIAN_BLUR_FILTER_SIZE_1 = 3;
	cv::GaussianBlur(mat_gray, mat_gaussian_blur_1, Size(GAUSSIAN_BLUR_FILTER_SIZE_1, GAUSSIAN_BLUR_FILTER_SIZE_1), 2);
	Mat mat_gaussian_blur_2;
	int GAUSSIAN_BLUR_FILTER_SIZE_2 = 21;
	cv::GaussianBlur(mat_gray, mat_gaussian_blur_2, Size(GAUSSIAN_BLUR_FILTER_SIZE_2, GAUSSIAN_BLUR_FILTER_SIZE_2), 20);
	//将两幅高斯模糊图像做减法，得到一幅包含边缘点的结果图像
	cv::absdiff(mat_gaussian_blur_1, mat_gaussian_blur_2, mat_dst);
	//TODO::下面逻辑待尝试
// 	if ( normalize && radius1 != radius2 ) {
// 		int[] pixels = null;
// 		int max = 0;
// 		for ( int y = 0; y < height; y++ ) {
// 			pixels = getRGB( image2, 0, y, width, 1, pixels );
// 			for ( int x = 0; x < width; x++ ) {
// 				int rgb = pixels[x];
// 				int r = (rgb >> 16) & 0xff;
// 				int g = (rgb >> 8) & 0xff;
// 				int b = rgb & 0xff;
// 				if ( r > max )
// 					max = r;
// 				if ( g > max )
// 					max = g;
// 				if ( b > max )
// 					max = b;
// 			}
// 		}
// 
// 		for ( int y = 0; y < height; y++ ) {
// 			pixels = getRGB( image2, 0, y, width, 1, pixels );
// 			for ( int x = 0; x < width; x++ ) {
// 				int rgb = pixels[x];
// 				int r = (rgb >> 16) & 0xff;
// 				int g = (rgb >> 8) & 0xff;
// 				int b = rgb & 0xff;
// 				r = r * 255 / max;
// 				g = g * 255 / max;
// 				b = b * 255 / max;
// 				pixels[x] = (rgb & 0xff000000) | (r << 16) | (g << 8) | b;
// 			}
// 			setRGB( image2, 0, y, width, 1, pixels );
// 		}
// 
// 	}

	return 0;
}
//max( |P1-P5|, |P2-P6|, |P3-P7|, |P4-P8| )
void CBusin_OpenCV_Filter_Tool_Inst::difference_Edge_Detect(const Mat& mat_src_gray, cv::Mat& mat_dst_gray, const cv::Rect& rect)
{
	//TODO::当Mat行与行直接不是连续的，是否有问题？
	// processing start and stop X,Y positions
	int startX  = rect.x + 1;
	int startY  = rect.y + 1;
	int stopX   = startX + rect.width - 2;
	int stopY   = startY + rect.height - 2;
	mat_dst_gray = Mat(mat_src_gray.size(), CV_8UC1);

	const int dstStride = mat_dst_gray.step[0];

	const int srcStride = mat_src_gray.step[0];

	const int dstOffset = dstStride - rect.width + 2;
	const int srcOffset = srcStride - rect.width + 2;

	int d = 0, max = 0;

	// data pointers
	const uchar* src = mat_src_gray.data;
	uchar* dst = mat_dst_gray.data;

	// allign pointers
	src += srcStride * startY + startX;
	dst += dstStride * startY + startX;

	// for each line
	for ( int y = startY; y < stopY; y++ )
	{
		// for each pixel
		for ( int x = startX; x < stopX; x++, src++, dst++ )
		{
			// left diagonal
			max = (int) src[-srcStride - 1] - src[srcStride + 1];
			if ( max < 0 )
				max = -max;

			// right diagonal
			d = (int) src[-srcStride + 1] - src[srcStride - 1];
			if ( d < 0 )
				d = -d;
			if ( d > max )
				max = d;
			// vertical
			d = (int) src[-srcStride] - src[srcStride];
			if ( d < 0 )
				d = -d;
			if ( d > max )
				max = d;
			// horizontal
			d = (int) src[-1] - src[1];
			if ( d < 0 )
				d = -d;
			if ( d > max )
				max = d;

			*dst = (uchar) max;
		}
		src += srcOffset;
		dst += dstOffset;
	}

	// draw black rectangle to remove those pixels, which were not processed
	// (this needs to be done for those cases, when filter is applied "in place" -
	// source image is modified instead of creating new copy)
	cv::rectangle( mat_dst_gray, rect, Scalar(0));
}

int CBusin_OpenCV_Filter_Tool_Inst::photocopy_gimp(const Mat& mat_src_gray, const SPhotocopy_Vals& photocopy_vals, Mat& mat_gray_result)
{
	CV_Assert(mat_src_gray.type() == CV_8UC1);
	//原图中待处理的起始位置
	const int nX_start = 0, nY_start = 0;
	const int nWidth = mat_src_gray.cols; //原图宽
	const int nHeight = mat_src_gray.rows; //原图高
	int nElem_size = mat_src_gray.elemSize();//每个像素所占字节数
	bool has_alpha = false;//是否具有alpha通道

	uchar* pDest1 = NULL;//pDest1 for blur radius
	CNewBuffMngr<uchar> buff_mng_dest1(pDest1, nWidth * nHeight);
	int progress = 0;
	const int nMax_progress = nWidth * nHeight * 3;

	for (int row = 0; row < nHeight; row++)
	{
		//获取行首地址
		const uchar *ptr_src = mat_src_gray.ptr<uchar>(row);
		uchar * ptr_dest = pDest1 + (0 - nY_start) * nWidth + (0 - nX_start) + row * nWidth;
		for (int col = 0; col < nWidth; col++)
		{
			/* desaturate */
			ptr_dest[col] = (uchar) ptr_src[col * nElem_size];

			/* compute  transfer */
			double dVal = pow (ptr_dest[col], (1.0 / GAMMA));
			ptr_dest[col] = (uchar) boost::algorithm::clamp(dVal, 0, 255);
		}
	}


	/*  Calculate the standard deviations  from blur and mask radius */
	double  dRadius = MAX(1.0, 10 * (1.0 - photocopy_vals.sharpness));
	dRadius   = fabs (dRadius) + 1.0;
	double dStd_dev1 = sqrt (-(dRadius * dRadius) / (2 * log (1.0 / 255.0)));

	dRadius   = fabs (photocopy_vals.mask_radius) + 1.0;
	double dStd_dev2 = sqrt (-(dRadius * dRadius) / (2 * log (1.0 / 255.0)));
	/*  derive the constants for calculating the gaussian from the std dev  */
	double       n_p1[5], n_m1[5];
	double       n_p2[5], n_m2[5];
	double       d_p1[5], d_m1[5];
	double       d_p2[5], d_m2[5];
	double       bd_p1[5], bd_m1[5];
	double       bd_p2[5], bd_m2[5];

	find_constants(n_p1, n_m1, d_p1, d_m1, bd_p1, bd_m1, dStd_dev1);
	find_constants(n_p2, n_m2, d_p2, d_m2, bd_p2, bd_m2, dStd_dev2);

	/*  First the vertical pass  */
	double* pVp1 = NULL, *pVp2 = NULL, *pVm1 = NULL, *pVm2 = NULL;
	int          initial_p1[4];
	int          initial_p2[4];
	int          initial_m1[4];
	int          initial_m2[4];
	//给相关变量分配内存空间（采用内存管理对象来管理内存）
	double* pVal_p1 = NULL;
	CNewBuffMngr<double> buff_mng_val_p1(pVal_p1, MAX (nWidth, nHeight));
	double* pVal_p2 = NULL;
	CNewBuffMngr<double> buff_mng_val_p2(pVal_p2, MAX (nWidth, nHeight));
	double* pVal_m1 = NULL;
	CNewBuffMngr<double> buff_mng_val_m1(pVal_m1, MAX (nWidth, nHeight));
	double* pVal_m2 = NULL;
	CNewBuffMngr<double> buff_mng_val_m2(pVal_m2, MAX (nWidth, nHeight));
	uchar* pDest2 = NULL;//dest2 for mask radius
	CNewBuffMngr<uchar> buff_mng_dest2(pDest2, nWidth * nHeight);
	//先按列后按行计算
	for (int col = 0; col < nWidth; col++)
	{
		memset(pVal_p1, 0, nHeight * sizeof (double));
		memset(pVal_p2, 0, nHeight * sizeof (double));
		memset(pVal_m1, 0, nHeight * sizeof (double));
		memset(pVal_m2, 0, nHeight * sizeof (double));

		uchar* pSrc1  = pDest1 + col;
		uchar* sp_p1 = pSrc1;
		uchar* sp_m1 = pSrc1 + (nHeight - 1) * nWidth;
		pVp1   = pVal_p1;
		pVp2   = pVal_p2;
		pVm1   = pVal_m1 + (nHeight - 1);
		pVm2   = pVal_m2 + (nHeight - 1);

		/*  Set up the first vals  */
		initial_p1[0] = sp_p1[0];
		initial_m1[0] = sp_m1[0];

		for (int row = 0; row < nHeight; row++)
		{
			double *vpptr1, *vmptr1;
			double *vpptr2, *vmptr2;

			int nTerms = (row < 4) ? row : 4;

			vpptr1 = pVp1; vmptr1 = pVm1;
			vpptr2 = pVp2; vmptr2 = pVm2;
			int i = 0;
			for ( i = 0; i <= nTerms; i++)
			{
				*vpptr1 += n_p1[i] * sp_p1[-i * nWidth] - d_p1[i] * pVp1[-i];
				*vmptr1 += n_m1[i] * sp_m1[i * nWidth] - d_m1[i] * pVm1[i];

				*vpptr2 += n_p2[i] * sp_p1[-i * nWidth] - d_p2[i] * pVp2[-i];
				*vmptr2 += n_m2[i] * sp_m1[i * nWidth] - d_m2[i] * pVm2[i];
			}

			for (int j = i; j <= 4; j++)
			{
				*vpptr1 += (n_p1[j] - bd_p1[j]) * initial_p1[0];
				*vmptr1 += (n_m1[j] - bd_m1[j]) * initial_m1[0];

				*vpptr2 += (n_p2[j] - bd_p2[j]) * initial_p1[0];
				*vmptr2 += (n_m2[j] - bd_m2[j]) * initial_m1[0];
			}

			sp_p1 += nWidth;
			sp_m1 -= nWidth;
			pVp1   += 1;
			pVp2   += 1;
			pVm1   -= 1;
			pVm2   -= 1;
		}

		transfer_pixels(pVal_p1, pVal_m1, pDest1 + col, nWidth, nHeight);
		transfer_pixels(pVal_p2, pVal_m2, pDest2 + col, nWidth, nHeight);
	}
	//先按行后案列计算
	for (int row = 0; row < nHeight; row++)
	{
		memset(pVal_p1, 0, nWidth * sizeof (double));
		memset(pVal_p2, 0, nWidth * sizeof (double));
		memset(pVal_m1, 0, nWidth * sizeof (double));
		memset(pVal_m2, 0, nWidth * sizeof (double));

		uchar* pSrc1 = pDest1 + row * nWidth;
		uchar* pSrc2 = pDest2 + row * nWidth;
		uchar* sp_p1 = pSrc1;
		uchar* sp_p2 = pSrc2;
		uchar* sp_m1 = pSrc1 + nWidth - 1;
		uchar* sp_m2 = pSrc2 + nWidth - 1;
		pVp1   = pVal_p1;
		pVp2   = pVal_p2;
		pVm1   = pVal_m1 + nWidth - 1;
		pVm2   = pVal_m2 + nWidth - 1;

		/*  Set up the first vals  */
		initial_p1[0] = sp_p1[0];
		initial_p2[0] = sp_p2[0];
		initial_m1[0] = sp_m1[0];
		initial_m2[0] = sp_m2[0];

		for (int col = 0; col < nWidth; col++)
		{
			double *vpptr1, *vmptr1;
			double *vpptr2, *vmptr2;

			int nTerms = (col < 4) ? col : 4;

			vpptr1 = pVp1; vmptr1 = pVm1;
			vpptr2 = pVp2; vmptr2 = pVm2;
			int i = 0;
			for (i = 0; i <= nTerms; i++)
			{
				*vpptr1 += n_p1[i] * sp_p1[-i] - d_p1[i] * pVp1[-i];
				*vmptr1 += n_m1[i] * sp_m1[i] - d_m1[i] * pVm1[i];

				*vpptr2 += n_p2[i] * sp_p2[-i] - d_p2[i] * pVp2[-i];
				*vmptr2 += n_m2[i] * sp_m2[i] - d_m2[i] * pVm2[i];
			}

			for (int j = i; j <= 4; j++)
			{
				*vpptr1 += (n_p1[j] - bd_p1[j]) * initial_p1[0];
				*vmptr1 += (n_m1[j] - bd_m1[j]) * initial_m1[0];

				*vpptr2 += (n_p2[j] - bd_p2[j]) * initial_p2[0];
				*vmptr2 += (n_m2[j] - bd_m2[j]) * initial_m2[0];
			}

			sp_p1 ++;
			sp_p2 ++;
			sp_m1 --;
			sp_m2 --;
			pVp1 ++;
			pVp2 ++;
			pVm1 --;
			pVm2 --;
		}

		transfer_pixels(pVal_p1, pVal_m1, pDest1 + row * nWidth, 1, nWidth);
		transfer_pixels(pVal_p2, pVal_m2, pDest2 + row * nWidth, 1, nWidth);
	}

	/* Compute the ramp value which sets 'pct_black' % of the darkened pixels black */
	double dRamp_down = compute_ramp(pDest1, pDest2, nWidth * nHeight, photocopy_vals.pct_black, 1, photocopy_vals.threshold);
	double dRamp_up   = compute_ramp(pDest1, pDest2, nWidth * nHeight, 1.0 - photocopy_vals.pct_white, 0, photocopy_vals.threshold);

	/* Initialize the pixel regions. */
	mat_gray_result = Mat(mat_src_gray.size(), CV_8UC1, Scalar(255));

	for (int row = 0; row < nHeight; row++)
	{
		const uchar* ptr_src  = mat_src_gray.ptr<uchar>(row);
		uchar* ptr_dest = mat_gray_result.ptr<uchar>(row);
		uchar  *ptr_blur = pDest1 + (0 - nY_start) * nWidth + (0 - nX_start) + row * nWidth;
		uchar  *ptr_avg  = pDest2 + (0 - nY_start) * nWidth + (0 - nX_start) + row * nWidth;
		for (int col = 0; col < nWidth; col++)
		{
			double  lightness = 0.0;
			if (ptr_avg[col] > EPSILON)
			{
				double diff = (double) ptr_blur[col] / (double) ptr_avg[col];
				double mult = 0.0;
				if (diff < photocopy_vals.threshold)
				{
					if (dRamp_down == 0.0)
					{
						mult = 0.0;
					}
					else
					{
						mult = (dRamp_down - MIN (dRamp_down, (photocopy_vals.threshold - diff))) / dRamp_down;
					}
					lightness = boost::algorithm::clamp(ptr_blur[col] * mult, 0, 255);
				}
				else
				{
					if (dRamp_up == 0.0)
					{
						mult = 1.0;
					}
					else
					{
						mult = MIN (dRamp_up, (diff - photocopy_vals.threshold)) / dRamp_up;
					}

					lightness = 255 - (1.0 - mult) * (255 - ptr_blur[col]);
					lightness = boost::algorithm::clamp(lightness, 0, 255);
				}
			}
			else
			{
				lightness = 0;
			}

			if (nElem_size < 3)
			{
				ptr_dest[col * nElem_size] = (uchar) lightness;
				if (has_alpha)
				{
					ptr_dest[col * nElem_size + 1] = ptr_src[col * nElem_size + 1];
				}
			}
			else
			{
				ptr_dest[col * nElem_size + 0] = lightness;
				ptr_dest[col * nElem_size + 1] = lightness;
				ptr_dest[col * nElem_size + 2] = lightness;

				if (has_alpha)
				{
					ptr_dest[col * nElem_size + 3] = ptr_src[col * nElem_size + 3];
				}
			}
		}
	}
	return 0;
}

//TODO::此算法还有待调整，目前一点效果都不行
int CBusin_OpenCV_Filter_Tool_Inst::photocopy_myself(const Mat& mat_src_gray, const SPhotocopy_Vals& photocopy_vals, Mat& mat_gray_result)
{
	//第一次均值滤波
	size_t nBlur_radius = photocopy_vals.mask_radius / 3;
	Mat mat_dest1_after_blur;//for nBlur_radius
	cv::blur(mat_src_gray, mat_dest1_after_blur, cv::Size(nBlur_radius, nBlur_radius));
	//第二次均值滤波
	Mat mat_dest2_after_blur;//for nMask_radius
	cv::GaussianBlur(mat_src_gray, mat_dest2_after_blur, cv::Size(photocopy_vals.mask_radius, photocopy_vals.mask_radius), 5);
	//计算dest1和dest2之间的Ramp_up和Ramp_down
	size_t nWidth = mat_src_gray.cols;
	size_t nHeight = mat_src_gray.rows;
	double ramp_down = compute_ramp(mat_dest1_after_blur.data, mat_dest2_after_blur.data, nWidth * nHeight, photocopy_vals.pct_black, 1, photocopy_vals.threshold);
	double ramp_up   = compute_ramp(mat_dest1_after_blur.data, mat_dest2_after_blur.data, nWidth * nHeight, 1.0 - photocopy_vals.pct_white, 0, photocopy_vals.threshold);
	mat_gray_result = Mat(mat_src_gray.size(), CV_8UC1, Scalar(255));
	for (int row = 0; row < nHeight; row++)
	{
		for (int col = 0; col < nWidth; col++)
		{
			double lightness = 0.0;
			if (mat_dest2_after_blur.data[col] > EPSILON)
			{
				double diff = (double) mat_dest1_after_blur.data[col] / (double) mat_dest2_after_blur.data[col];
				double mult = 0.0;
				if (diff < photocopy_vals.threshold)
				{
					if (ramp_down == 0.0)
						mult = 0.0;
					else
						mult = (ramp_down - MIN (ramp_down,(photocopy_vals.threshold - diff))) / ramp_down;
					lightness = boost::algorithm::clamp(mat_dest1_after_blur.data[col] * mult, 0, 255);
				}
				else
				{
					if (ramp_up == 0.0)
						mult = 1.0;
					else
						mult = MIN (ramp_up,(diff - photocopy_vals.threshold)) / ramp_up;

					lightness = 255 - (1.0 - mult) * (255 - mat_dest1_after_blur.data[col]);
					lightness = boost::algorithm::clamp(lightness, 0, 255);
				}
			}
			else
			{
				lightness = 0;
			}
			mat_gray_result.at<uchar>(row, col) = lightness;
		}
	}
	return 0;
}

void CBusin_OpenCV_Filter_Tool_Inst::transfer_pixels(double *pdScr1, double *pdSrc2, uchar *pcDest, int nJump, int nWidth)
{
	double dSum = 0.0;

	for(int i = 0; i < nWidth; i++)
	{
		dSum = pdScr1[i] + pdSrc2[i];
		if (dSum > 255) dSum = 255;
		else if(dSum < 0) dSum = 0;

		*pcDest = (uchar) dSum;

		pcDest += nJump;
	}
}
//使用高斯算子的四阶近似来实现非正式序列的常数
void CBusin_OpenCV_Filter_Tool_Inst::find_constants(double n_p[], double n_m[], double d_p[], double d_m[], double bd_p[], double bd_m[], double dStd_dev)
{
	int    i = 0;
	double constants[8] = {0};
	double nDiv = 0.0;

	/*  The constants used in the implementation of a casual sequence
	*  using a 4th order approximation of the gaussian operator
	*/

	nDiv = sqrt (2 * boost::math::constants::pi<double>()) * dStd_dev;
	constants [0] = -1.783 / dStd_dev;
	constants [1] = -1.723 / dStd_dev;
	constants [2] = 0.6318 / dStd_dev;
	constants [3] = 1.997  / dStd_dev;
	constants [4] = 1.6803 / nDiv;
	constants [5] = 3.735 / nDiv;
	constants [6] = -0.6803 / nDiv;
	constants [7] = -0.2598 / nDiv;

	n_p [0] = constants[4] + constants[6];
	n_p [1] = exp (constants[1]) *
		(constants[7] * sin (constants[3]) -
		(constants[6] + 2 * constants[4]) * cos (constants[3])) +
		exp (constants[0]) *
		(constants[5] * sin (constants[2]) -
		(2 * constants[6] + constants[4]) * cos (constants[2]));
	n_p [2] = 2 * exp (constants[0] + constants[1]) *
		((constants[4] + constants[6]) * cos (constants[3]) * cos (constants[2]) -
		constants[5] * cos (constants[3]) * sin (constants[2]) -
		constants[7] * cos (constants[2]) * sin (constants[3])) +
		constants[6] * exp (2 * constants[0]) +
		constants[4] * exp (2 * constants[1]);
	n_p [3] = exp (constants[1] + 2 * constants[0]) *
		(constants[7] * sin (constants[3]) - constants[6] * cos (constants[3])) +
		exp (constants[0] + 2 * constants[1]) *
		(constants[5] * sin (constants[2]) - constants[4] * cos (constants[2]));
	n_p [4] = 0.0;

	d_p [0] = 0.0;
	d_p [1] = -2 * exp (constants[1]) * cos (constants[3]) -
		2 * exp (constants[0]) * cos (constants[2]);
	d_p [2] = 4 * cos (constants[3]) * cos (constants[2]) * exp (constants[0] + constants[1]) +
		exp (2 * constants[1]) + exp (2 * constants[0]);
	d_p [3] = -2 * cos (constants[2]) * exp (constants[0] + 2 * constants[1]) -
		2 * cos (constants[3]) * exp (constants[1] + 2 * constants[0]);
	d_p [4] = exp (2 * constants[0] + 2 * constants[1]);

#ifndef ORIGINAL_READABLE_CODE
	memcpy(d_m, d_p, 5 * sizeof(double));
#else
	for (i = 0; i <= 4; i++)
		d_m [i] = d_p [i];
#endif

	n_m[0] = 0.0;
	for (i = 1; i <= 4; i++)
		n_m [i] = n_p[i] - d_p[i] * n_p[0];

	{
		double sum_n_p, sum_n_m, sum_d;
		double a, b;

		sum_n_p = 0.0;
		sum_n_m = 0.0;
		sum_d   = 0.0;

		for (i = 0; i <= 4; i++)
		{
			sum_n_p += n_p[i];
			sum_n_m += n_m[i];
			sum_d += d_p[i];
		}

#ifndef ORIGINAL_READABLE_CODE
		sum_d++;
		a = sum_n_p / sum_d;
		b = sum_n_m / sum_d;
#else
		a = sum_n_p / (1 + sum_d);
		b = sum_n_m / (1 + sum_d);
#endif

		for (i = 0; i <= 4; i++)
		{
			bd_p[i] = d_p[i] * a;
			bd_m[i] = d_m[i] * b;
		}
	}
}

double CBusin_OpenCV_Filter_Tool_Inst::compute_ramp(const uchar *pDest1, const uchar *pDest2, int nLength, double dPct_black, int nUnder_threshold, double dPhotocopy_threshold)
{
	int    hist[2000];
	int    count;
	int    i;
	int    sum;

	memset(hist, 0, sizeof (int) * 2000);
	count = 0;

	for (i = 0; i < nLength; i++)
	{
		if (*pDest2 != 0)
		{
			double diff = (double) *pDest1 / (double) *pDest2;

			if (nUnder_threshold)
			{
				if (diff < dPhotocopy_threshold)
				{
					hist[(int) (diff * 1000)] += 1;
					count += 1;
				}
			}
			else
			{
				if (diff >= dPhotocopy_threshold && diff < 2.0)
				{
					hist[(int) (diff * 1000)] += 1;
					count += 1;
				}
			}
		}

		pDest1++;
		pDest2++;
	}

	if (dPct_black == 0.0 || count == 0)
		return (nUnder_threshold ? 1.0 : 0.0);

	sum = 0;
	for (i = 0; i < 2000; i++)
	{
		sum += hist[i];
		if (((double) sum / (double) count) > dPct_black)
		{
			if (nUnder_threshold)
				return (dPhotocopy_threshold - (double) i / 1000.0);
			else
				return ((double) i / 1000.0 - dPhotocopy_threshold);
		}
	}

	return (nUnder_threshold ? 0.0 : 1.0);
}

CBusin_OpenCV_Filter_Tool_Inst CBusin_OpenCV_Filter_Tool_Inst::ms_inst;
