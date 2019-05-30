/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Inscribed_Polygon.h
* @brief: 求取内接多边形
* @author:	minglu2
* @version: 1.0
* @date: 2019/05/23
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/05/23	<td>minglu	<td>Create head file </tr>
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
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
//宏定义

//类型定义
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
		//查找最大面积的轮廓
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		cv::waitKey(0);
	}
	//仅基于中心扩展算法来获取最大内接矩形（此方法得当的结果并非全局最优）
	void test_max_inscribed_rect_only_Using_CEA()
	{
		const string str_img_path = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_for_MER/1.jpg";
		Mat mat_src_gray = imread(str_img_path, IMREAD_GRAYSCALE);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_gray, 100, 255,THRESH_BINARY_INV);
		cv::imshow("mat_src_gray", mat_src_gray);
		//查找最大面积的轮廓
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//计算最小外接矩形
		cv::RotatedRect rRect =  cv::minAreaRect(vec_max_area_contour);
		Rect rect_MER = InSquare(mat_src_gray, rRect.center);
		//在原图中画出矩形
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
		//查找最大面积的轮廓
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//计算轮廓的最小外包矩形
		Rect rect_bbox =  cv::boundingRect(vec_max_area_contour);
		Rect rect_MER = get_upRight_MER_using_traversing(mat_src_gray, rect_bbox);
		Time = (double)cvGetTickCount() - Time;
		printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//毫秒
		//在原图中画出矩形
		cv::rectangle(mat_src_gray, rect_MER, Scalar(0, 0, 0),1, LINE_8,0);

		std::cout << __FUNCTION__ << " | MER:" << rect_MER << ", area:" << rect_MER.width * rect_MER.height << endl;
		cv::imshow("mat src with MER", mat_src_gray);
		cv::waitKey(0);
	}
	void test_max_inscribed_rect_using_traversing2()
	{
		double Time = (double)cvGetTickCount();
		const string str_img_path = "./images_for_MER/2.jpg";
		Mat mat_src_gray = imread(str_img_path, IMREAD_GRAYSCALE);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_gray, 100, 255,THRESH_BINARY_INV);
		imwrite(str_img_path+"_binary_gray.jpg", mat_src_gray);
		cv::imshow("mat_src_gray", mat_src_gray);
		//查找最大面积的轮廓
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//计算轮廓的最小外包矩形
		Rect rect_bbox =  cv::boundingRect(vec_max_area_contour);
		Rect rect_MER = get_upRight_MER_using_traversing2(mat_src_gray, vec_max_area_contour, rect_bbox);
		Time = (double)cvGetTickCount() - Time;
		printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//毫秒
		//在原图中画出矩形
		cv::rectangle(mat_src_gray, rect_MER, Scalar(100, 100, 100), 1, LINE_8,0);
		std::cout << __FUNCTION__ << " | MER:" << rect_MER << ", area:" << rect_MER.width * rect_MER.height << endl;
		cv::imshow("mat src with MER", mat_src_gray);
		cv::imwrite(str_img_path + "_withMER.jpg", mat_src_gray);
		cv::waitKey(0);
	}
	void test_rect()
	{
		const string str_img_path = "./images_for_MER/2.jpg_binary_gray.jpg";
		Mat mat_src_gray = imread(str_img_path);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		//在原图中画出矩形
		Rect rect_IOR(194, 140, 137, 139); //正确
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
		vector<Point> biggestContour = FindBiggestContour(src);
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

	//最大内接圆实例
	void test_max_inscribed_circle()
	{
		Mat src = imread("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_for_MER/cloud.png");
		Mat temp;
		cvtColor(src,temp, COLOR_BGR2GRAY);
		threshold(temp,temp, 100, 255,THRESH_OTSU);
		imshow("src",temp);
		//寻找最大轮廓
		vector<Point>  VPResult = FindBiggestContour(temp);
		if (VPResult.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//寻找最大内切圆
		//在目标轮廓中查找x的最大值和y的最大值
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
		//绘制结果
		circle(src,center,maxdist,Scalar(0,0,255));
		imshow("dst",src);
		waitKey();
	}
protected:
	int get_MER(const Mat& mat_src, string& str_err_reason)
	{
		//将图像转换为灰度图
		//查找轮廓
		return 0;
	}
	//图像为黑底
	vector<Point> FindBiggestContour(const Mat& mat_src_gray)
	{    
		int nCount = 0; //代表最大轮廓的序号
		double dMax_area_contour = -1; //代表最大轮廓的面积大小
		std::vector<std::vector<cv::Point>>contours;
		findContours(mat_src_gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		if (contours.empty())
		{
			printf("%s | Number of contours is 0\n", __FUNCTION__);
			return vector<Point>();
		}
		for (int i = 0; i < contours.size(); ++i)
		{
			double dTemp_area =  contourArea(contours[i]);//这里采用的是轮廓大小
			if (dMax_area_contour < dTemp_area )
			{
				nCount = i;
				dMax_area_contour = dTemp_area;
			}
#ifdef _DEBUG
			Mat drawing = Mat::zeros( mat_src_gray.size(), CV_8UC1 );
			drawContours( drawing, contours, i, Scalar(255), 2, 8);
			cv::imshow("drawing", drawing);
#endif // _DEBUG
		}
		return contours[nCount];
	}

	/**
	* @brief 求取连通区域内接矩
	* @param img:输入图像，单通道二值图，深度为8
	* @param center:最小外接矩的中心
	* @return  最大内接矩形
	* 基于中心扩展算法
	*/
	cv::Rect InSquare(Mat &img, const Point center)
	{
		// --[1]参数检测
		if(img.empty() || img.channels() > 1 || img.depth() != CV_8U)
		{
			printf("%s | error.\n", __FUNCTION__);
			return Rect();
		}
		//[1]

		// --[2] 初始化变量
		int edge[4] = { 0 };
		edge[0] = center.y + 1;//top
		edge[1] = center.x + 1;//right
		edge[2] = center.y - 1;//bottom
		edge[3] = center.x - 1;//left
		//[2]

		// --[3]边界扩展(中心扩散法)
		bool EXPAND[4] ={ 1, 1, 1, 1};//扩展标记位
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
	* @brief expandEdge 扩展边界函数
	* @param img:输入图像，单通道二值图，深度为8
	* @param edge  边界数组，存放4条边界值
	* @param edgeID 当前边界号
	* @return 布尔值 确定当前边界是否可以扩展
	*/
	bool expandEdge(const Mat & img, int edge[], const int edgeID)
	{
		//[1] --初始化参数
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
							//计算面积
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
	Rect get_upRight_MER_using_traversing2(const Mat& mat_src_binary_gray, const vector<Point>& contour,const Rect& rect_bbox)
	{
		double dMax_area = INT_MIN;
		int flag_x_min = 0, flag_x_max = 0, flag_y_min = 0, flag_y_max = 0;
		int nXmin = rect_bbox.x, nXmax = rect_bbox.x + rect_bbox.width - 1;
		int nYmin = rect_bbox.y, nYmax = rect_bbox.y + rect_bbox.height - 1;
		int nMin_dist_X = 2; //X方向两边界的最小间隔
		int nMin_dist_Y = 2; //Y方向上两边界的最小间隔
		for (int i = nXmin; i <= nXmax; ++i)
		{
			for (int j = i + nMin_dist_X; j <= nXmax; ++j)
			{
				for (int m = nYmin; m <= nYmax; ++m)
				{
					//判定三条线所得的两个顶点是否在轮廓外
					//根据灰度值来判定点是否在轮廓外
// 					if (mat_src_binary_gray.at<uchar>(m, i) == 0 && mat_src_binary_gray.at<uchar>(m, i + 1) != 0
// 						&& mat_src_binary_gray.at<uchar>(m + 1, i) != 0 && mat_src_binary_gray.at<uchar>(m + 1, i + 1) != 0)
// 					{//至少有一个点在轮廓外
// 						continue;
// 					}
// 					if (mat_src_binary_gray.at<uchar>(m, j) == 0 && mat_src_binary_gray.at<uchar>(m, j - 1) != 0
// 						&& mat_src_binary_gray.at<uchar>(m +1, j) != 0 && mat_src_binary_gray.at<uchar>(m + 1, j - 1) != 0)
// 					{
// 						continue;
// 					}
					for (int n = m + nMin_dist_Y; n <= nYmax; ++n)
					{
// 						if (mat_src_binary_gray.at<uchar>(n, i) == 0 && mat_src_binary_gray.at<uchar>(n, i + 1) != 0
// 							&& mat_src_binary_gray.at<uchar>(n - 1, i) != 0 && mat_src_binary_gray.at<uchar>(n - 1, i + 1) != 0)
// 						{//至少有一个点在轮廓外
// 							continue;
// 						}
// 						if (mat_src_binary_gray.at<uchar>(n, j) == 0 && mat_src_binary_gray.at<uchar>(n, j - 1) != 0
// 							&& mat_src_binary_gray.at<uchar>(n - 1, j) != 0 && mat_src_binary_gray.at<uchar>(n - 1, j - 1) != 0)
// 						{
// 							continue;
// 						}
						if (no_black_in_rect(mat_src_binary_gray, i, j, m, n) == true)
						{
							//计算面积
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
						{//有黑点在四边所组成的矩形内,则不再当前方向上继续扩展边界
							break;
						}
					}
				}
			}
		}
		std::cout << __FUNCTION__ << " | flag_x_min:" << flag_x_min << ", flag_x_max:" << flag_x_max 
			<< ", flag_y_min:" << flag_y_min << ",flag_y_max:" << flag_y_max << ", dMax_area:" << dMax_area << endl;
		return Rect(flag_x_min, flag_y_min, flag_x_max - flag_x_min + 1, flag_y_max - flag_y_min + 1);
	}
	//判定矩形四边是否含有黑色点，任一边含有黑色点都返回真
	bool rect_edge_has_black(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		//上边
		int y = nYmin;
		int x = 0;
		for ( x = nXmin; x <= nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return true;
			}
		}
		//右边
		x = nXmax;
		for (y = nYmin; y <= nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return true;
			}
		}
		//下边
		y = nYmax;
		for ( x = nXmin; x <= nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return true;
			}
		}
		//左边
		x = nXmin;
		for (y = nYmin; y <= nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return true;
			}
		}
		return false;
	}
	//矩形内无黑色点
	bool no_black_in_rect(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		//上边
		int y = nYmin + 1;
		int x = 0;
		for ( x = nXmin + 1; x < nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return false;
			}
		}
		//右边
		x = nXmax - 1;
		for (y = nYmin + 1; y < nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return false;
			}
		}
		//下边
		y = nYmax - 1;
		for ( x = nXmin + 1; x < nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return false;
			}
		}
		//左边
		x = nXmin + 1;
		for (y = nYmin + 1; y < nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//为黑色
				return false;
			}
		}
		return true;
	}
	//点在轮廓外
	bool is_out_of_contour(const Mat& mat_src_binary_gray, const vector<Point>& contour, const Point& p0)
	{
		if (contour.empty())
		{
			printf("%s | contour is empty.", __FUNCTION__);
			exit(-1);
		}
		//获取轮廓内的一个坐标点
		//取轮廓0、1/3点的中心点
		Point temp0_0 = contour[0];
		Point temp1_3 =  contour[contour.size()/3];
		Point point_middle_1 = (temp0_0 + temp1_3) / 2;
		//取轮廓1/2、2/3处的点的中心点
		Point temp1_2 = contour[contour.size()/2];
		Point temp2_3 = contour[contour.size() * 2 / 3];
		Point point_middle_2 = (temp1_2 + temp2_3) / 2;
		//再取中心点的中心点，所得点肯定在轮廓内
		Point p1 = (point_middle_1 + point_middle_2) / 2 ;
		//从点p0到点p1：像素值先黑后白
		if (mat_src_binary_gray.at<uchar>(p0) == 0 && mat_src_binary_gray.at<uchar>(p1) != 0)
		{
			return true;
		}
		else
		{
			false;
		}
	}
private:
};
//函数原型定义
