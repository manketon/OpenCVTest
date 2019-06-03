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
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
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
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
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
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
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
	void test_max_inscribed_rect_using_traversing2()
	{
		double Time = (double)cvGetTickCount();
		const string str_img_path = "./images_for_MER/2.jpg";
		Mat mat_src_gray = imread(str_img_path, IMREAD_GRAYSCALE);
		cv::namedWindow("mat_src_gray", CV_WINDOW_AUTOSIZE);
		threshold(mat_src_gray, mat_src_gray, 100, 255,THRESH_BINARY_INV);
		imwrite(str_img_path+"_binary_gray.jpg", mat_src_gray);
		cv::imshow("mat_src_gray", mat_src_gray);
		//����������������
		vector<Point> vec_max_area_contour = FindBiggestContour(mat_src_gray);
		if (vec_max_area_contour.empty())
		{
			printf("%s | error\n", __FUNCTION__);
			return;
		}
		//������������С�������
		Rect rect_bbox =  cv::boundingRect(vec_max_area_contour);
		Rect rect_sub(231, 178, 69, 84);
		Rect rect_MER = get_upRight_MER_using_traversing2(mat_src_gray, vec_max_area_contour, rect_bbox, rect_sub);
		Time = (double)cvGetTickCount() - Time;
		printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//����
		//��ԭͼ�л�������
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

	//����ڽ�Բʵ��
	void test_max_inscribed_circle()
	{
		Mat src = imread("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_for_MER/cloud.png");
		Mat temp;
		cvtColor(src,temp, COLOR_BGR2GRAY);
		threshold(temp,temp, 100, 255,THRESH_OTSU);
		imshow("src",temp);
		//Ѱ���������
		vector<Point>  VPResult = FindBiggestContour(temp);
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
	vector<Point> FindBiggestContour(const Mat& mat_src_gray)
	{    
		int nCount = 0; //����������������
		double dMax_area_contour = -1; //������������������С
		std::vector<std::vector<cv::Point>>contours;
		findContours(mat_src_gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		if (contours.empty())
		{
			printf("%s | Number of contours is 0\n", __FUNCTION__);
			return vector<Point>();
		}
		for (int i = 0; i < contours.size(); ++i)
		{
			double dTemp_area =  contourArea(contours[i]);//������õ���������С
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
	BG_Element** x_flag_ptr_Arr; //x_flag_ptr_Arr[i]������ŵı�ڵ��ʾ�������ڣ�x=i�У��߶�((i,begin),��i,end��)֮�䶼Ϊ��ɫ
	BG_Element** y_flag_ptr_Arr;
	//���Ӿ��κ���С��Ӿ���֮������������ڽӾ���
	Rect get_upRight_MER_using_traversing2(const Mat& mat_src_binary_gray, const vector<Point>& contour,const Rect& rect_bbox
		, const Rect& rect_sub)
	{
		double dMax_area = INT_MIN;
		int flag_x_min = 0, flag_x_max = 0, flag_y_min = 0, flag_y_max = 0;
		int nXmin = rect_bbox.x, nXmax = rect_bbox.x + rect_bbox.width - 1;
		int nYmin = rect_bbox.y, nYmax = rect_bbox.y + rect_bbox.height - 1;
		int nSub_rect_MinX = rect_sub.x, nSub_rect_MaxX = rect_sub.x + rect_sub.width - 1;
		int nSub_rect_MinY = rect_sub.y, nSub_rect_MaxY = rect_sub.y + rect_sub.height - 1;
		//����һ������
		x_flag_ptr_Arr = new BG_Element*[mat_src_binary_gray.cols];
		memset(x_flag_ptr_Arr, 0, sizeof(BG_Element*) * mat_src_binary_gray.cols);
		y_flag_ptr_Arr = new BG_Element*[mat_src_binary_gray.rows];
		memset(y_flag_ptr_Arr, 0, sizeof(BG_Element*) * mat_src_binary_gray.rows);
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
					if (mat_src_binary_gray.at<uchar>(y + 1, x) != 0)
					{//��һ����Ϊ�׵�
						point_last_black_before_white = Point(x, y);
						has_begin = true;
					}
					else if (has_begin== true && mat_src_binary_gray.at<uchar>(y - 1, x) != 0)
					{//��һ����Ϊ�׵�
						point_first_black_after_white = Point(x, y);
						has_end = true;
					}

					if (has_begin == true && has_end == true)
					{
						BG_Element* p = new BG_Element;
						p->nBegin = point_last_black_before_white.y;
						p->nEnd = point_first_black_after_white.y;
						p->pNext = x_flag_ptr_Arr[x];
						x_flag_ptr_Arr[x] = p;
						has_begin = false;
						has_end = false;
						//����һ��
						--y;
					}
				}
			}
		}
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
					if (mat_src_binary_gray.at<uchar>(y, x + 1) != 0)
					{//��һ�е�Ϊ�׵�
						point_last_black_before_white = Point(x, y);
						has_begin = true;
					}
					else if (has_begin == true && mat_src_binary_gray.at<uchar>(y, x - 1) != 0)
					{//ǰһ�е�Ϊ�׵�
						point_first_black_after_white = Point(x, y);
						has_end = true;
					}
					if (has_begin == true && has_end == true)
					{
						BG_Element* p = new BG_Element;
						p->nBegin = point_last_black_before_white.x;
						p->nEnd = point_first_black_after_white.x;
						p->pNext = y_flag_ptr_Arr[y];
						y_flag_ptr_Arr[y] = p;
						has_begin = false;
						has_end = false;
						//����һ��
						--x;
					}
				}
			}
		}

		int nMin_dist_X = 2; //X�������߽����С��������ֵ���ɹ���
		int nMin_dist_Y = 2; //Y���������߽����С���
		for (int i = nXmin; i <= nXmax/* && i <= nSub_rect_MinX*/; ++i)
		{
			for (int j = i + nMin_dist_X/* nSub_rect_MaxX*/; j <= nXmax; ++j)
			{
				for (int m = nYmin; m <= nYmax/* && m <= nSub_rect_MinY*/; ++m)
				{
					//�ж����������õ����������Ƿ���������
					//���ݻҶ�ֵ���ж����Ƿ���������
					if (mat_src_binary_gray.at<uchar>(m, i) == 0 && mat_src_binary_gray.at<uchar>(m, i + 1) != 0
						&& mat_src_binary_gray.at<uchar>(m + 1, i) != 0 && mat_src_binary_gray.at<uchar>(m + 1, i + 1) != 0)
					{//������һ������������
						continue;
					}
					if (mat_src_binary_gray.at<uchar>(m, j) == 0 && mat_src_binary_gray.at<uchar>(m, j - 1) != 0
						&& mat_src_binary_gray.at<uchar>(m +1, j) != 0 && mat_src_binary_gray.at<uchar>(m + 1, j - 1) != 0)
					{
						continue;
					}
					for (int n = m + nMin_dist_Y/*nSub_rect_MaxY*/; n <= nYmax; ++n)
					{
						if (mat_src_binary_gray.at<uchar>(n, i) == 0 && mat_src_binary_gray.at<uchar>(n, i + 1) != 0
							&& mat_src_binary_gray.at<uchar>(n - 1, i) != 0 && mat_src_binary_gray.at<uchar>(n - 1, i + 1) != 0)
						{//������һ������������
							continue;
						}
						if (mat_src_binary_gray.at<uchar>(n, j) == 0 && mat_src_binary_gray.at<uchar>(n, j - 1) != 0
							&& mat_src_binary_gray.at<uchar>(n - 1, j) != 0 && mat_src_binary_gray.at<uchar>(n - 1, j - 1) != 0)
						{
							continue;
						}
					    //���ڴ�ʱ�߽粻�Ǵ���̾��뿪ʼ�ģ��п��ܻ������м�ĺڵ㣬���´���
						if (no_black_in_rect2(mat_src_binary_gray, i, j, m, n) == true)
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
		//�ͷ��ڴ�
		for (int i = 0; i != mat_src_binary_gray.cols; ++i)
		{
			BG_Element* p = x_flag_ptr_Arr[i];
			while (p != NULL)
			{
				x_flag_ptr_Arr[i] = p->pNext;
				delete p;
				p = x_flag_ptr_Arr[i];
			}
		}
		delete [] x_flag_ptr_Arr;
		for (int i = 0; i != mat_src_binary_gray.rows; ++i)
		{
			BG_Element* p = y_flag_ptr_Arr[i];
			while (p != NULL)
			{
				y_flag_ptr_Arr[i] = p->pNext;
				delete p;
				p = y_flag_ptr_Arr[i];
			}
		}
		delete [] y_flag_ptr_Arr;
		std::cout << __FUNCTION__ << " | flag_x_min:" << flag_x_min << ", flag_x_max:" << flag_x_max 
			<< ", flag_y_min:" << flag_y_min << ",flag_y_max:" << flag_y_max << ", dMax_area:" << dMax_area << endl;
		return Rect(flag_x_min, flag_y_min, flag_x_max - flag_x_min + 1, flag_y_max - flag_y_min + 1);
	}
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
	/************************************
	* Method:    no_black_in_rect
	* Brief:  �ж��������Ƿ��޺�ɫ��
	* Access:    protected 
	* Returns:   bool
	* Qualifier: Ŀ���ֵ�Ҷ�ͼ������ʱ�������������ڶ�Ϊ��ɫ�޺�ɫ��
	             ˼�룺������ε��ڲ�����Ϻ��к�ɫ�㣬������ڲ��кڵ㣻���򣬾����ڲ�϶��޺�ɫ�㣨������ԭ��
	*Parameter: const Mat & mat_src_binary_gray -[in/out]  Ŀ���ֵ�Ҷ�ͼ������ʱ�������������ڶ�Ϊ��ɫ�޺�ɫ
	*Parameter: int nXmin -[in/out]  
	*Parameter: int nXmax -[in/out]  
	*Parameter: int nYmin -[in/out]  
	*Parameter: int nYmax -[in/out]  
	************************************/
	bool no_black_in_rect(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		//�ϱ�
		int y = nYmin + 1;
		int x = 0;
		for ( x = nXmin + 1; x < nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return false;
			}
		}
		//�ұ�
		x = nXmax - 1;
		for (y = nYmin + 1; y < nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return false;
			}
		}
		//�±�
		y = nYmax - 1;
		for ( x = nXmin + 1; x < nXmax; ++x)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return false;
			}
		}
		//���
		x = nXmin + 1;
		for (y = nYmin + 1; y < nYmax; ++y)
		{
			if (mat_src_binary_gray.at<uchar>(y, x) == 0)
			{//Ϊ��ɫ
				return false;
			}
		}
		return true;
	}
	bool no_black_in_rect2(const Mat& mat_src_binary_gray, int nXmin, int nXmax, int nYmin, int nYmax)
	{
		
		BG_Element * p = NULL;
		//�ϱ�((nXmin, nYmin + 1), (nXmax, nYmin + 1))֮�����޺ڵ�
		p = y_flag_ptr_Arr[nYmin + 1];
		bool bNo_black = no_black_on_line(p, nXmin, nXmax);
		if (!bNo_black)
		{
			return false;
		}
		//�ұ�
		p = x_flag_ptr_Arr[nXmax - 1];
		bNo_black = no_black_on_line(p, nYmin, nYmax);
		if (!bNo_black)
		{
			return false;
		}
		
		//�±�
		p = y_flag_ptr_Arr[nYmax - 1];
		bNo_black = no_black_on_line(p, nXmin, nXmax);
		if (!bNo_black)
		{
			return false;
		}
		//���
		p = x_flag_ptr_Arr[nXmin + 1];
		bNo_black = no_black_on_line(p, nYmin, nYmax);
		if (!bNo_black)
		{
			return false;
		}
		return true;
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
private:
};
//����ԭ�Ͷ���
