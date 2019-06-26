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
using namespace cv;
using namespace std;
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
		Mat mat_dst_gray;
		Mat mat_src_gray;
		cv::cvtColor(mat_src_bgr, mat_src_gray, CV_BGR2GRAY);
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

void CBusin_OpenCV_Filter_Tool_Inst::difference_Edge_Detect(const Mat& mat_src_gray, cv::Mat& mat_dst_gray, const cv::Rect& rect)
{
	//TODO::当Mat行与行直接不是连续的，是否有问题？
	// processing start and stop X,Y positions
	int startX  = rect.x + 1;
	int startY  = rect.y + 1;
	int stopX   = startX + rect.width - 2;
	int stopY   = startY + rect.height - 2;
	mat_dst_gray = Mat(mat_src_gray.size(), CV_8UC1);

	int dstStride = mat_dst_gray.cols * mat_dst_gray.channels();
	int dst_yushu = dstStride % 4;
	if (dst_yushu != 0)
	{
		dstStride +=  4 - dst_yushu;
	}

	int srcStride = mat_src_gray.cols * mat_src_gray.channels();
	int src_yushu = srcStride % 4;
	if (src_yushu != 0)
	{
		srcStride += 4 - src_yushu;
	}

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

CBusin_OpenCV_Filter_Tool_Inst CBusin_OpenCV_Filter_Tool_Inst::ms_inst;
