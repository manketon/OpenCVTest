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
	static CInscribed_Polygon_Tool& instace()
	{
		static CInscribed_Polygon_Tool obj;
		return obj;
	}
	void test_max_inscribed_rect()
	{
		const string str_img_path = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_for_MER/1.jpg";
		Mat mat_src_gray = imread(str_img_path, IMREAD_GRAYSCALE);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_gray, 100, 255,THRESH_BINARY_INV);
		cv::imshow("mat_src_gray", mat_src_gray);
		//����������������
		vector<Point> vec_max_area_contour;
		int ret = FindBiggestContour(mat_src_gray, vec_max_area_contour);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		cv::waitKey(0);
	}
	//������������չ�㷨����ȡ����ڽӾ��Σ��˷����õ��Ľ������ȫ�����ţ�
	void test_max_inscribed_rect_only_Using_CEA()
	{
		const string str_img_path = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_for_MER/1.jpg";
		Mat mat_src_gray = imread(str_img_path, IMREAD_GRAYSCALE);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_gray, 100, 255,THRESH_BINARY_INV);
		cv::imshow("mat_src_gray", mat_src_gray);
		//����������������
		vector<Point> vec_max_area_contour;
		int ret = FindBiggestContour(mat_src_gray, vec_max_area_contour);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//������С��Ӿ���
		cv::RotatedRect rRect =  cv::minAreaRect(vec_max_area_contour);
		Rect rect_MER = InSquare(mat_src_gray, rRect.center);
		//��ԭͼ�л�������
		cv::rectangle(mat_src_gray, rect_MER, Scalar(0, 0, 0),1, LINE_8,0);
		mat_src_gray.at<uchar>(rRect.center) = 0;

		std::cout << __FUNCTION__ << " | MER:" << rect_MER << endl;
		Rect rect_MER_MY(224, 138, 168, 79);
		cv::rectangle(mat_src_gray, rect_MER_MY, Scalar(190, 0, 0),1, LINE_8,0);
		cv::imshow("mat src with MER", mat_src_gray);
		cv::waitKey(0);
	}

	void test_max_inscribed_rect_using_traversing()
	{
		double Time = (double)cvGetTickCount();
		const string str_img_path = "./images_for_MER/2.jpg";
		Mat mat_src_gray = imread(str_img_path, IMREAD_GRAYSCALE);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_gray, 100, 255,THRESH_BINARY_INV);
		cv::imshow("mat_src_gray", mat_src_gray);
		//����������������
		vector<Point> vec_max_area_contour;
		int ret = FindBiggestContour(mat_src_gray, vec_max_area_contour);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//������������С�������
		Rect rect_bbox =  cv::boundingRect(vec_max_area_contour);
		Rect rect_MER = get_upRight_MER_using_traversing(mat_src_gray, rect_bbox);
		Time = (double)cvGetTickCount() - Time;
		printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//����
		//��ԭͼ�л�������
		cv::rectangle(mat_src_gray, rect_MER, Scalar(0, 0, 0),1, LINE_8,0);

		std::cout << __FUNCTION__ << " | MER:" << rect_MER << ", area:" << rect_MER.width * rect_MER.height << endl;
		cv::imshow("mat src with MER", mat_src_gray);
		cv::waitKey(0);
	}
	
	void test_max_inscribed_rect_using_traversing_for_rotated()
	{
		double Time = (double)cvGetTickCount();
		const string str_img_path = "./images_for_MER/2_rotated.jpg";
		Mat mat_src_bgr = imread(str_img_path, IMREAD_COLOR);
		//��ԭͼת��Ϊ�Ҷ�ͼ
		Mat mat_src_gray;
		cvtColor(mat_src_bgr, mat_src_gray, COLOR_BGR2GRAY);
		//����������������
		//���ֵ�ҶȾ���(��ɫΪ��)
		Mat mat_src_binary_gray;
		threshold(mat_src_gray, mat_src_binary_gray, 100, 255,THRESH_BINARY_INV);
		cv::imshow("mat_src_binary_gray", mat_src_binary_gray);
		//���������ڽӾ��ο϶����������������У����Ȳ���������������
		//����������������
		vector<Point> vec_max_area_contour;
		int ret = FindBiggestContour(mat_src_binary_gray, vec_max_area_contour);
		if (ret)
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		RotatedRect rRect_min_area = cv::minAreaRect(vec_max_area_contour);
		Mat mat_with_max_area;
		Rect rect_MER_with_max_area;
		double dDegree_with_max_area = 0;
		for (double dDegree = 0; dDegree <= 90; dDegree += 1)
		{
			Mat mat_rotated = CBusin_Opencv_Transform_Tool::instance().rotate_image_without_loss(
				mat_src_bgr, rRect_min_area.center, dDegree, 1, Scalar(255, 255, 255));

			Rect rect_sub(231, 178, 69, 84);
			Rect rect_MER;
			int ret = get_upRight_MER_using_traversing2(str_img_path, mat_rotated, rect_sub, rect_MER);
			if (ret)
			{
				std::cout << __FUNCTION__ << " | failed, ret:" << ret << endl;
			}
			if (rect_MER.area() > rect_MER_with_max_area.area())
			{
				mat_with_max_area = mat_rotated.clone();
				rect_MER_with_max_area = rect_MER;
				dDegree_with_max_area = dDegree;
			}
//			cv::waitKey(0);
		}
		Time = (double)cvGetTickCount() - Time;
		printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//����

		//��������Ӧ�ľ����л�������
		cv::rectangle(mat_with_max_area, rect_MER_with_max_area, Scalar(255, 255, 255), 1, LINE_8,0);
		std::cout << __FUNCTION__ << " | MER:" << rect_MER_with_max_area << ", area:" << rect_MER_with_max_area.area() 
			<< ", dDegree_with_max_area:" << dDegree_with_max_area << endl;
		//��ʱ������������������ת��ע�⣺��ʱͼƬ�ķֱ��ʷ����˱仯������Ӧ������ת������ĵ�����ת��
// 		Mat mat_withMER = CBusin_Opencv_Transform_Tool::instance().rotate_image_without_loss(
// 			mat_with_max_area, rRect_min_area.center/*���ĵ���ת����*/, -1 * dDegree_with_max_area, 1);
// 		cv::imshow("mat src with MER", mat_withMER);
// 		cv::imwrite(str_img_path + "_withMER.jpg", mat_withMER);
		cv::imshow("mat_max_area with MER", mat_with_max_area);
		cv::imwrite(str_img_path + "_withMER.jpg", mat_with_max_area);
		cv::waitKey(0);
	}
	//���ںڵװ���ͼƬ��ͨ����ԭͼ��������ת�õ���ת���ͼ��Ȼ�����ת����ͼƬ������ٲ���������ڽӾ���
	//����������ڽӾ��μ�¼�������������ս������ԭͼ��Ӧ������ڽӾ���
	void test_max_inscribed_rect_using_traversing_for_rotated2();
	void func_thread_upRight_MER(const Mat& mat_src_bgr, const Point2f& center, double dDegree, const Rect& rect_sub_src
		, Mat& mat_rotated, Rect& rect_upRight_MER)
	{
		std::cout << __FUNCTION__ << std::endl;
		//��ȡ��ת���ͼƬ���Ӿ���
		Rect rect_sub_dst_shrink;//����ת���Ӿ��ε���С���Σ��Ա�֤��϶�������������ڽ�upRight������
		mat_rotated = CBusin_Opencv_Transform_Tool::instance().rotate_image_without_loss(
			mat_src_bgr, center, dDegree, 1, rect_sub_src, rect_sub_dst_shrink, Scalar(0, 0, 0));
		//������ת����ͼƬ���Ӿ�����ȡ����������ڽ�upRight����
		int ret = get_upRight_MER_using_traversing3(mat_rotated, rect_sub_dst_shrink, rect_upRight_MER);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | failed, ret:" << ret << endl;
		}
	}
	void test_max_inscribed_rect_using_traversing2()
	{
		double Time = (double)cvGetTickCount();
		const string str_img_path = "./images_for_MER/2.jpg";
		Mat mat_src_bgr = imread(str_img_path, IMREAD_COLOR);


		Rect rect_sub(231, 178, 69, 84);
		Rect rect_MER;
		int ret = get_upRight_MER_using_traversing2(str_img_path, mat_src_bgr, rect_sub, rect_MER);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | failed, ret:" << ret << endl;
		}
		Time = (double)cvGetTickCount() - Time;
		printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//����
		//ע��Ӧ���ںڵ�ͼ�л�������
		cv::rectangle(mat_src_bgr, rect_MER, Scalar(100, 100, 100), 1, LINE_8,0);
		std::cout << __FUNCTION__ << " | MER:" << rect_MER << ", area:" << rect_MER.width * rect_MER.height << endl;
		cv::imshow("mat src gray with MER", mat_src_bgr);
		cv::imwrite(str_img_path + "_withMER.jpg", mat_src_bgr);
		cv::waitKey(0);
	}
	void test_rect()
	{
		const string str_img_path = "./images_for_MER/2.jpg_binary_gray.jpg";
		Mat mat_src_gray = imread(str_img_path);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		//��ԭͼ�л�������
		Rect rect_IOR(194, 140, 137, 139); //��ȷ
		cv::rectangle(mat_src_gray, Point(194, 140), Point(330, 278), Scalar(100, 100, 100), 1, LINE_8,0);
		cv::imshow("mat src with RECT", mat_src_gray);
		cv::imwrite(str_img_path + "_withRect.jpg", mat_src_gray);
		cv::waitKey(0);
	}
	int test3()
	{
		/// Create an image
		const int r = 100;
		Mat src = Mat::zeros( Size( 4*r, 4*r ), CV_8U );
		/// Create a sequence of points to make a contour
		vector<Point2f> vert(6);
		vert[0] = Point( 3*r/2, static_cast<int>(1.34*r) );
		vert[1] = Point( 1*r, 2*r );
		vert[2] = Point( 3*r/2, static_cast<int>(2.866*r) );
		vert[3] = Point( 5*r/2, static_cast<int>(2.866*r) );
		vert[4] = Point( 3*r, 2*r );
		vert[5] = Point( 5*r/2, static_cast<int>(1.34*r) );
		/// Draw it in src
		for( int i = 0; i < 6; i++ )
		{
			line( src, vert[i],  vert[(i+1)%6], Scalar( 255 ), 3 );
		}
		/// Get the contours
		vector<vector<Point> > contours;
		findContours( src, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
		/// Calculate the distances to the contour
		Mat raw_dist( src.size(), CV_32F );
		for( int i = 0; i < src.rows; i++ )
		{
			for( int j = 0; j < src.cols; j++ )
			{
				raw_dist.at<float>(i,j) = (float)pointPolygonTest( contours[0], Point2f((float)j, (float)i), true );
			}
		}
		double minVal, maxVal;
		minMaxLoc( raw_dist, &minVal, &maxVal );
		minVal = abs(minVal);
		maxVal = abs(maxVal);
		/// Depicting the  distances graphically
		Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
		for( int i = 0; i < src.rows; i++ )
		{
			for( int j = 0; j < src.cols; j++ )
			{
				if( raw_dist.at<float>(i,j) < 0 )
				{
					drawing.at<Vec3b>(i,j)[0] = (uchar)(255 - abs(raw_dist.at<float>(i,j)) * 255 / minVal);
				}
				else if( raw_dist.at<float>(i,j) > 0 )
				{
					drawing.at<Vec3b>(i,j)[2] = (uchar)(255 - raw_dist.at<float>(i,j) * 255 / maxVal);
				}
				else
				{
					drawing.at<Vec3b>(i,j)[0] = 255;
					drawing.at<Vec3b>(i,j)[1] = 255;
					drawing.at<Vec3b>(i,j)[2] = 255;
				}
			}
		}
		//get the biggest Contour
		vector<Point> biggestContour;
		int ret = FindBiggestContour(src, biggestContour);
		if (ret)
		{
			std::cout <<__FUNCTION__ << " | failed, ret:" << ret << std::endl;
		}
		//find the maximum enclosed circle 
		int dist = 0;
		int maxdist = 0;
		Point center;
		for(int i=0;i<src.cols;i++)
		{
			for(int j=0;j<src.rows;j++)
			{
				dist = pointPolygonTest(biggestContour,cv::Point(i,j),true);
				if(dist>maxdist)
				{
					maxdist=dist;
					center=cv::Point(i,j);
				}
			}
		}
		circle(drawing,center,maxdist,Scalar(255,255,255));
		/// Show your results
		imshow( "Source", src );
		imshow( "Distance and maximum enclosed circle", drawing );
		waitKey();
		return 0;
	}

	//����ڽ�Բʵ��
	void test_max_inscribed_circle()
	{
		Mat src = imread("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_for_MER/cloud.png");
		Mat temp;
		cvtColor(src,temp, COLOR_BGR2GRAY);
		threshold(temp,temp, 100, 255,THRESH_OTSU);
		imshow("src",temp);
		//Ѱ���������
		vector<Point>  VPResult; 
		int ret = FindBiggestContour(temp, VPResult);
		if (VPResult.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//Ѱ���������Բ
		//��Ŀ�������в���x�����ֵ��y�����ֵ
		int nMin_X = INT_MAX, nMax_X = INT_MIN, nMin_Y = INT_MAX, nMax_Y = INT_MIN;
		for (int i = 0; i != VPResult.size(); ++i)
		{
			if (VPResult[i].x > nMax_X)
			{
				nMax_X =  VPResult[i].x;
			}
			else if (VPResult[i].x < nMin_X)
			{
				nMin_X = VPResult[i].x;
			}
			if (VPResult[i].y > nMax_Y)
			{
				nMax_Y = VPResult[i].y;
			}
			else if (VPResult[i].y < nMin_Y)
			{
				nMin_Y = VPResult[i].y;
			}
		}
		double maxdist = 0;
		Point center;
		for(int i = nMin_X; i < nMax_X; ++i)
		{
			for(int j = nMin_Y; j < nMax_Y; ++j)
			{
				double dDist = pointPolygonTest(VPResult,cv::Point(i,j), true);
				if(dDist > maxdist)
				{
					maxdist=dDist;
					center=cv::Point(i,j);
				}
			}
		}
		//���ƽ��
		circle(src,center,maxdist,Scalar(0,0,255));
		imshow("dst",src);
		waitKey();
	}
protected:
	int get_MER(const Mat& mat_src, string& str_err_reason)
	{
		//��ͼ��ת��Ϊ�Ҷ�ͼ
		//��������
		return 0;
	}
	//ͼ��Ϊ�ڵ�
	int FindBiggestContour(const Mat& mat_src_gray, vector<Point>& vec_max_area_contour)
	{    
		int nCount = 0; //����������������
		double dMax_area_contour = -1; //������������������С
		std::vector<std::vector<cv::Point>>contours;
		findContours(mat_src_gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		if (contours.empty())
		{//δ�ҵ�����
			printf("%s | Number of contours is 0\n", __FUNCTION__);
			return -1;
		}
		for (int i = 0; i < contours.size(); ++i)
		{
			//��������Խ��棬������׼ȷ
			double dTemp_area =  contourArea(contours[i]);//������õ���������С
			if (dMax_area_contour < dTemp_area )
			{
				nCount = i;
				dMax_area_contour = dTemp_area;
			}
#if 0
			Mat drawing = Mat::zeros( mat_src_gray.size(), CV_8UC1 );
			drawContours( drawing, contours, i, Scalar(255), 2, 8);
			cv::imshow("drawing", drawing);
#endif // _DEBUG
		}
		vec_max_area_contour = contours[nCount];
		return 0;
	}

	/**
	* @brief ��ȡ��ͨ�����ڽӾ�
	* @param img:����ͼ�񣬵�ͨ����ֵͼ�����Ϊ8
	* @param center:��С��Ӿص�����
	* @return  ����ڽӾ���
	* ����������չ�㷨
	*/
	cv::Rect InSquare(Mat &img, const Point center)
	{
		// --[1]�������
		if(img.empty() || img.channels() > 1 || img.depth() != CV_8U)
		{
			printf("%s | error.\n", __FUNCTION__);
			return Rect();
		}
		//[1]

		// --[2] ��ʼ������
		int edge[4] = { 0 };
		edge[0] = center.y + 1;//top
		edge[1] = center.x + 1;//right
		edge[2] = center.y - 1;//bottom
		edge[3] = center.x - 1;//left
		//[2]

		// --[3]�߽���չ(������ɢ��)
		bool EXPAND[4] ={ 1, 1, 1, 1};//��չ���λ
		int n = 0;
		while (EXPAND[0]||EXPAND[1]||EXPAND[2]||EXPAND[3])
		{
			int edgeID = n%4;
			EXPAND[edgeID] = expandEdge(img,edge,edgeID);
			n++;
		}
		//[3]

		std::cout << __FUNCTION__ << " | edge0:"<< edge[0] << ", edge1:" << edge[1] 
		<< ", edge2:" << edge[2] << ", edge3:"<< edge[3] << std::endl;
		Point tl=Point(edge[3],edge[0]);
		Point br=Point(edge[1],edge[2]);
		return Rect(tl,br);
	}

	/**
	* @brief expandEdge ��չ�߽纯��
	* @param img:����ͼ�񣬵�ͨ����ֵͼ�����Ϊ8
	* @param edge  �߽����飬���4���߽�ֵ
	* @param edgeID ��ǰ�߽��
	* @return ����ֵ ȷ����ǰ�߽��Ƿ������չ
	*/
	bool expandEdge(const Mat & img, int edge[], const int edgeID)
	{
		//[1] --��ʼ������
		int nc=img.cols;
		int nr=img.rows;

		switch (edgeID) {
		case 0:
			if(edge[0]>nr)
				return false;
			for(int i=edge[3];i<=edge[1];++i)
			{
				if(img.at<uchar>(edge[0],i)==0)
					return false;
			}
			edge[0]++;
			return true;
			break;
		case 1:
			if(edge[1]>nc)
				return false;
			for(int i=edge[2];i<=edge[0];++i)
			{
				if(img.at<uchar>(i,edge[1])==0)
					return false;
			}
			edge[1]++;
			return true;
			break;
		case 2:
			if(edge[2]<0)
				return false;
			for(int i=edge[3];i<=edge[1];++i)
			{
				if(img.at<uchar>(edge[2],i)==0)
					return false;
			}
			edge[2]--;
			return true;
			break;
		case 3:
			if(edge[3]<0)
				return false;
			for(int i=edge[2];i<=edge[0];++i)
			{
				if(img.at<uchar>(i,edge[3])==0)
					return false;
			}
			edge[3]--;
			return true;
			break;
		default:
			return false;
			break;
		}
	}
	Rect get_upRight_MER_using_traversing(const Mat& mat_src_binary_gray,const Rect& rect_bbox)
	{
		double dMax_area = INT_MIN;
		int flag_x_min = 0, flag_x_max = 0, flag_y_min = 0, flag_y_max = 0;
		int nXmin = rect_bbox.x, nXmax = rect_bbox.x + rect_bbox.width;
		int nYmin = rect_bbox.y, nYmax = rect_bbox.y + rect_bbox.height;
		for (int i = nXmin + 1; i < nXmax; ++i)
		{
			for (int j = i + 1; j < nXmax; ++j)
			{
				for (int m = nYmin + 1; m < nYmax; ++m)
				{
					for (int n = m + 1; n < nYmax; ++n)
					{
						if (rect_edge_has_black(mat_src_binary_gray, i, j, m, n) == false)
						{
							//�������
							double dArea_tmp = (j - i)*(n - m);
							if (dMax_area < dArea_tmp)
							{
								dMax_area = dArea_tmp;
								flag_x_min = i;
								flag_x_max = j;
								flag_y_min = m;
								flag_y_max = n;
							}
						}
					}
				}
			}
		}
		std::cout << __FUNCTION__ << " | flag_x_min:" << flag_x_min << ", flag_x_max:" << flag_x_max 
			<< ", flag_y_min:" << flag_y_min << ",flag_y_max:" << flag_y_max << endl;
		return Rect(flag_x_min, flag_y_min, flag_x_max - flag_x_min, flag_y_max - flag_y_min);
	}

	//���Ӿ��κ���С��Ӿ���֮������������ڽӾ���
	int get_upRight_MER_using_traversing2(const string&str_img_path, const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_MER)
	{
		if (mat_src_bgr.type() != CV_8UC3)
		{
			std::cout << __FUNCTION__ << " | type of mat_src_bgr:" << mat_src_bgr.type() << ", is not CV_8UC3:" << CV_8UC3 << std::endl; 
			return -1;
		}
		//��BRGͼƬת��Ϊ�Ҷ�ͼ
		Mat mat_src_gray;
		cv::cvtColor(mat_src_bgr, mat_src_gray, COLOR_BGR2GRAY);
		//���ֵ�ҶȾ���(��ɫΪ��)
		Mat mat_src_binary_gray;
		cv::namedWindow("mat_src_binary_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_binary_gray, 100, 255,THRESH_BINARY_INV);
//		imwrite(str_img_path+"_binary_gray.jpg", mat_src_binary_gray);
		cv::imshow("mat_src_binary_gray", mat_src_binary_gray);
		//���������ڽӾ��ο϶����������������У����Ȳ���������������
		//����������������
		vector<Point> vec_max_area_contour;
		int ret = FindBiggestContour(mat_src_binary_gray, vec_max_area_contour);
		if (ret)
		{
			printf("%s | error\n", __FUNCTION__);
			return -1;
		}

		//������������������Ӧ����С�������
		Rect rect_bbox =  cv::boundingRect(vec_max_area_contour);
		if (rect_bbox.empty() || rect_bbox.height <= 0 || rect_bbox.width <= 0)
		{
			std::cout << __FUNCTION__ << " | failed to get bbox, rect_bbox:" << rect_bbox << endl; 
			return -1;
		}
		double dMax_area = INT_MIN;
		int flag_x_min = 0, flag_x_max = 0, flag_y_min = 0, flag_y_max = 0;
		//����С��Ӿ��εı߽���չ���ֵ��Ϊx��y��ȡֵ��Χ��ע�⣺��С��Ӿ��εı߽���ܺ�������Ե�غϣ���Ҫ��չ
		int nXmin = rect_bbox.x - 1, nXmax = rect_bbox.x + rect_bbox.width - 1 + 1; 
		int nYmin = rect_bbox.y - 1, nYmax = rect_bbox.y + rect_bbox.height - 1 + 1;
		int nSub_rect_MinX = rect_sub.x, nSub_rect_MaxX = rect_sub.x + rect_sub.width - 1; 
		int nSub_rect_MinY = rect_sub.y, nSub_rect_MaxY = rect_sub.y + rect_sub.height - 1;
		BG_Element** x_white_line_ptr_Arr = NULL; //x_white_line_ptr_Arr[i]������ŵı�ڵ��ʾ�������ڣ�x=i�У��߶�[(i,begin),��i,end)]֮�䶼Ϊ��ɫ
		BG_Element** y_white_line_ptr_Arr = NULL;

		//������ֵ�ҶȾ��󣬷ֱ��ȡX/Y�����ϵİ�ɫ�߶�����
		ret = get_white_line_arr(mat_src_binary_gray, x_white_line_ptr_Arr, y_white_line_ptr_Arr, nXmin, nXmax, nYmin, nYmax);
		int nMin_dist_X = 2; //X�������߽����С���
		int nMin_dist_Y = 2; //Y���������߽����С���
		for (int i = nXmin; i <= nXmax/* && i <= nSub_rect_MinX*/; ++i)
		{
			for (int j = i + nMin_dist_X /*nSub_rect_MaxX*/; j <= nXmax; ++j)
			{
				for (int m = nYmin; m <= nYmax/* && m <= nSub_rect_MinY*/; ++m)
				{
					//�ж����������õ����������Ƿ���������
					//���ݻҶ�ֵ���ж����Ƿ���������
					if (mat_src_binary_gray.at<uchar>(m, i) == 0)
					{//��������
						continue;
					}
					if (mat_src_binary_gray.at<uchar>(m, j) == 0)
					{
						continue;
					}
					for (int n = m + nMin_dist_Y/* nSub_rect_MaxY*/; n <= nYmax; ++n)
					{
						if (mat_src_binary_gray.at<uchar>(n, i) == 0)
						{//��������
							continue;
						}
						if (mat_src_binary_gray.at<uchar>(n, j) == 0)
						{
							continue;
						}
					    //ʹ��rect_edge_has_black�ɹ�
						if (rect_edge_has_black2(mat_src_binary_gray, x_white_line_ptr_Arr, y_white_line_ptr_Arr, i, j, m, n) == false)
						{
							//�������
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
						{//�кڵ����ı�����ɵľ�����,���ٵ�ǰ�����ϼ�����չ�߽�
							break;
						}
					}
				}
			}
		}
		ret = release_white_line_arr(mat_src_binary_gray.size(), x_white_line_ptr_Arr, y_white_line_ptr_Arr);
		std::cout << __FUNCTION__ << " | flag_x_min:" << flag_x_min << ", flag_x_max:" << flag_x_max 
			<< ", flag_y_min:" << flag_y_min << ",flag_y_max:" << flag_y_max << ", dMax_area:" << dMax_area << endl;
		rect_MER =  Rect(flag_x_min, flag_y_min, flag_x_max - flag_x_min + 1, flag_y_max - flag_y_min + 1);
		return 0;
	}

	int get_upRight_MER_using_traversing3(const Mat& mat_src_bgr, const Rect& rect_sub, Rect& rect_upRight_MER);


	//�ж������ı��Ƿ��к�ɫ�㣬��һ�ߺ��к�ɫ�㶼������
	bool rect_edge_has_black(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		//�ϱ�
		int y = nYmin;
		int x = 0;
		for ( x = nXmin; x <= nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return true;
			}
		}
		//�ұ�
		x = nXmax;
		for (y = nYmin; y <= nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return true;
			}
		}
		//�±�
		y = nYmax;
		for ( x = nXmin; x <= nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return true;
			}
		}
		//���
		x = nXmin;
		for (y = nYmin; y <= nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return true;
			}
		}
		return false;
	}
	//�ж������ı��Ƿ��к�ɫ�㣬��һ�ߺ��к�ɫ�㶼������
	bool rect_edge_has_black2(const Mat& mat_src_binary_gray, BG_Element**&x_white_line_ptr_Arr, BG_Element**&y_while_line_ptr_Arr
		, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		BG_Element * p = NULL;
		//�ϱ�[(nXmin, nYmin), (nXmax, nYmin)]֮�����޺ڵ�
		p = y_while_line_ptr_Arr[nYmin];
		bool bNo_black = no_black_on_line(p, nXmin , nXmax);
		if (!bNo_black)
		{
			return true;
		}
		//�ұ�
		p = x_white_line_ptr_Arr[nXmax];
		bNo_black = no_black_on_line(p, nYmin, nYmax);
		if (!bNo_black)
		{
			return true;
		}

		//�±�
		p = y_while_line_ptr_Arr[nYmax];
		bNo_black = no_black_on_line(p, nXmin, nXmax);
		if (!bNo_black)
		{
			return true;
		}
		//���
		p = x_white_line_ptr_Arr[nXmin];
		bNo_black = no_black_on_line(p, nYmin, nYmax);
		if (!bNo_black)
		{
			return true;
		}
		return false;
	}

	//��x��yȷ�����ж���nBegin,nEnd��֮���Ƿ��кڵ�
	bool no_black_on_line(const BG_Element * p, size_t nBegin, size_t nEnd)
	{
		while (p != NULL)
		{
			if (nBegin >= p->nBegin && nEnd <= p->nEnd)
			{//�ҵ�ĳ������������������˵���߶�֮�䲻���к�ɫ
				return true;
			}
			else
			{
				p = p->pNext;
			}
		}
		//δ�ҵ�
		return false;
	}
	//����������
	bool is_out_of_contour(const Mat& mat_src_binary_gray, const vector<Point>& contour, const Point& p0)
	{
		if (contour.empty())
		{
			printf("%s | contour is empty.", __FUNCTION__);
			exit(-1);
		}
		//��ȡ�����ڵ�һ�������
		//ȡ����0��1/3������ĵ�
		Point temp0_0 = contour[0];
		Point temp1_3 =  contour[contour.size()/3];
		Point point_middle_1 = (temp0_0 + temp1_3) / 2;
		//ȡ����1/2��2/3���ĵ�����ĵ�
		Point temp1_2 = contour[contour.size()/2];
		Point temp2_3 = contour[contour.size() * 2 / 3];
		Point point_middle_2 = (temp1_2 + temp2_3) / 2;
		//��ȡ���ĵ�����ĵ㣬���õ�϶���������
		Point p1 = (point_middle_1 + point_middle_2) / 2 ;
		//�ӵ�p0����p1������ֵ�Ⱥں��
		if (mat_src_binary_gray.at<uchar>(p0) == 0 && mat_src_binary_gray.at<uchar>(p1) != 0)
		{
			return true;
		}
		else
		{
			false;
		}
	}
	int get_white_line_arr(const Mat& mat_src_binary_gray, BG_Element**& x_white_line_ptr_Arr, BG_Element**& y_white_line_ptr_Arr, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		//����һ������
		x_white_line_ptr_Arr = new BG_Element*[mat_src_binary_gray.cols];
		memset(x_white_line_ptr_Arr, 0, sizeof(BG_Element*) * mat_src_binary_gray.cols);
		y_white_line_ptr_Arr = new BG_Element*[mat_src_binary_gray.rows];
		memset(y_white_line_ptr_Arr, 0, sizeof(BG_Element*) * mat_src_binary_gray.rows);
		for (int x = nXmin; x <= nXmax; ++x)
		{
			Point point_last_black_before_white; 
			Point point_first_black_after_white;
			bool has_begin = false; //һ����ɫ�߶ε�ͷ�������
			bool has_end = false;//һ����ɫ�߶ε�β�������
			//ע�⣺һ���߶ε�β�������Ϊ��һ���߶ε�ͷ����
			for (int y = nYmin; y <= nYmax; ++y)
			{
				//�����������б���
				if (mat_src_binary_gray.at<uchar>(y, x) == 0)
				{//��ǰ��Ϊ�ڵ�
					if (mat_src_binary_gray.at<uchar>(y + 1, x) != 0 && false == has_begin)
					{//ͷ���δ��������һ����Ϊ�׵㣬��ǰ��϶�Ϊͷ
						point_last_black_before_white = Point(x, y);
						has_begin = true;
					}
					else if (mat_src_binary_gray.at<uchar>(y - 1, x) != 0 && true == has_begin)
					{//ͷ����������һ����Ϊ�׵㣬��ǰ��϶�Ϊβ��
						point_first_black_after_white = Point(x, y);
						has_end = true;
					}

					if (has_begin == true && has_end == true)
					{
						BG_Element* p = new BG_Element;
						p->nBegin = point_last_black_before_white.y + 1;
						p->nEnd = point_first_black_after_white.y - 1;
						p->pNext = x_white_line_ptr_Arr[x];
						x_white_line_ptr_Arr[x] = p;
						has_begin = false;
						has_end = false;
						//����һ��
						--y;
					}
				}
			}
		}
#ifdef PRINT_WHITE_LINE_ARR  //��ӡ����ɫ�߶ζ�Ӧ������
		for (int i = 0; i != mat_src_binary_gray.cols; ++i)
		{
			BG_Element* p = x_white_line_ptr_Arr[i];
			while (p != NULL)
			{
				std::cout << "col:" << i << ", [" << p->nBegin << "," << p->nEnd << "] all white" << endl;
				p = p->pNext;
			}
		}
#endif // PRINT_WHITE_LINE_ARR
		for (int y = nYmin; y <= nYmax; ++y)
		{
			Point point_last_black_before_white; 
			Point point_first_black_after_white;
			bool has_begin = false; //һ����ɫ�߶ε�ͷ�������
			bool has_end = false;//һ����ɫ�߶ε�β�������
			for (int x = nXmin; x <= nXmax; ++x)
			{
				//�����������б���
				if (mat_src_binary_gray.at<uchar>(y, x) == 0)
				{
					if (mat_src_binary_gray.at<uchar>(y, x + 1) != 0 && false == has_begin)
					{//��һ�е�Ϊ�׵���ͷδ���֣���������Ϊͷ
						point_last_black_before_white = Point(x, y);
						has_begin = true;
					}
					else if (mat_src_binary_gray.at<uchar>(y, x - 1) != 0 && true == has_begin)
					{//ǰһ�е�Ϊ�׵���ͷ������
						point_first_black_after_white = Point(x, y);
						has_end = true;
					}
					if (has_begin == true && has_end == true)
					{
						BG_Element* p = new BG_Element;
						p->nBegin = point_last_black_before_white.x + 1;
						p->nEnd = point_first_black_after_white.x - 1;
						p->pNext = y_white_line_ptr_Arr[y];
						y_white_line_ptr_Arr[y] = p;
						has_begin = false;
						has_end = false;
						//����һ��
						--x;
					}
				}
			}
		}
#ifdef PRINT_WHITE_LINE_ARR  //��ӡ����ɫ�߶ζ�Ӧ������
		for (int i = 0; i != mat_src_binary_gray.rows; ++i)
		{
			BG_Element* p = y_white_line_ptr_Arr[i];
			while (p != NULL)
			{
				std::cout << "row:" << i << ", [" << p->nBegin << "," << p->nEnd << "] all white" << endl;
				p = p->pNext;
			}
		}
#endif // PRINT_WHITE_LINE_ARR
		return 0;
	}
	int release_white_line_arr(const cv::Size& size_of_binary_gray_img, BG_Element**& x_white_line_ptr_Arr, BG_Element**& y_white_line_ptr_Arr)
	{
		//�ͷ��ڴ�
		for (int i = 0; i != size_of_binary_gray_img.width; ++i)
		{
			BG_Element* p = x_white_line_ptr_Arr[i];
			while (p != NULL)
			{
				x_white_line_ptr_Arr[i] = p->pNext;
				delete p;
				p = x_white_line_ptr_Arr[i];
			}
		}
		delete [] x_white_line_ptr_Arr;
		for (int i = 0; i != size_of_binary_gray_img.height; ++i)
		{
			BG_Element* p = y_white_line_ptr_Arr[i];
			while (p != NULL)
			{
				y_white_line_ptr_Arr[i] = p->pNext;
				delete p;
				p = y_white_line_ptr_Arr[i];
			}
		}
		delete [] y_white_line_ptr_Arr;
		return 0;
	}
private:
};
//����ԭ�Ͷ���
