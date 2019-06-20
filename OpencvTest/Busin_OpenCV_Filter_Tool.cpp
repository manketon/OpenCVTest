/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Filter_Tool.cpp
* @brief: ���˵���ļ����ܡ���; (Comment)��
* @author:	minglu2
* @version: 1.0
* @date: 2019/06/19
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
*  <tr> <td>1.0	    <td>2019/06/19	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "Busin_OpenCV_Filter_Tool.h"
#include "ImageMaster/include/ImageMaster.h"
#include <string>
#include "Boost_Common_Tool.h"
#include <iostream>
#include <boost/filesystem.hpp>
using namespace cv;
using namespace std;
CBusin_OpenCV_Filter_Tool_Inst& CBusin_OpenCV_Filter_Tool_Inst::instance()
{
	return ms_inst;
}
//����
int CBusin_OpenCV_Filter_Tool_Inst::test_sketch()
{
	string str_test_imgs_dir = "C:/Users/dell.dell-PC/Desktop/�˾�����ͼƬ/����/test_input";
	string str_output_dir = "C:/Users/dell.dell-PC/Desktop/�˾�����ͼƬ/����/test_output";
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
		//ȥɫ
		cvtColor(src, gray0, CV_BGR2GRAY);
		//��ɫ
		addWeighted(gray0, -1, NULL, 0, 255, gray1);
		//��˹ģ��,��˹�˵�Size������Ч���й�
		int nKernel_size = 11;
		GaussianBlur(gray1, gray1, Size(nKernel_size, nKernel_size), 0);
		imshow("result of gauss", gray1);
		//�ںϣ���ɫ����
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
		imshow("����", img_for_show);
		//��ֵ��
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

//�·���ɫȥ���ˣ��������鲿����PS��ͬ
int CBusin_OpenCV_Filter_Tool_Inst::test_Laplacian_sketch()
{
	std::string str_img_path = "C:/Users/dell.dell-PC/Desktop/�˾�����ͼƬ/����/lhj.jpg";
	Mat src = imread(str_img_path,1);
	int width = src.cols;
	int heigh = src.rows;
	Mat gray0, gray1;
	//ȥɫ
	cvtColor(src, gray0, CV_BGR2GRAY);
	const int MEDIAN_BLUR_FILTER_SIZE = 3;
//	cv::medianBlur(gray0, gray0, MEDIAN_BLUR_FILTER_SIZE);
	GaussianBlur(gray0, gray0, Size(MEDIAN_BLUR_FILTER_SIZE, MEDIAN_BLUR_FILTER_SIZE), 0);
	cv::Mat edges;
	//��Ե���
	const int LAPLACIAN_FILTER_SIZE = 3;
	cv::Laplacian(gray0, edges, CV_8U, LAPLACIAN_FILTER_SIZE);
	imshow("edges", edges);
	cv::Mat mask;
	const int EDGE_THRESHOLD = 10;
	cv::threshold(edges, mask, EDGE_THRESHOLD, 255, cv::THRESH_BINARY_INV);
	imshow("����",mask);
	waitKey();
	imwrite(str_img_path + "_result.jpg", mask);
	return 0;
}

//�߷����
int CBusin_OpenCV_Filter_Tool_Inst::test_GaoFanChaBaoLiu()
{
	int R = 11;
	std::string str_img_path = "C:/Users/dell.dell-PC/Desktop/�˾�����ͼƬ/����/lhj.jpg";
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

	//��ͨ�˲�����

	Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
		-1,  5, -1,
		0, -1,  0);
	Mat dstF;
	filter2D(img,dstF,img.depth(),kern);
	imshow("kernel",dstF);

	waitKey();
	//imwrite("D:/�߷����.jpg",dst);
	//	imwrite("D:/��ͨ�˲�.jpg",dstF);
	imwrite(str_img_path + "_result.jpg", dst);
	return 0;
}

int CBusin_OpenCV_Filter_Tool_Inst::test_photocopy()
{
	Mat src = imread("C:/Users/dell.dell-PC/Desktop/�˾�����ͼƬ/����/src.jpg", 1);
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

CBusin_OpenCV_Filter_Tool_Inst CBusin_OpenCV_Filter_Tool_Inst::ms_inst;
