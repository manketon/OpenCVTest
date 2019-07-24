#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "CBusin_Opencv_Transform_Tool.h"
#include "CBusin_OpenCV_Vector_Tool.h"
#include "Busin_OpenCV_Common_Tool.h"
#include "Busin_OpenCV_File_Tool.h"
#include "Busin_Opencv_Contour.h"
#include "Busin_OpenCV_Inscribed_Polygon.h"
#include "Busin_OpenCV_Filter_Tool.h"
#include "Busin_OpenCV_Connected_Graph.h"
using namespace std;
using namespace cv;

int test(const cv::Mat& src_mat)
{
	cv::Point center_point;
	center_point.x = src_mat.cols/2 + 0.5;
	center_point.y = src_mat.rows/2 + 0.5;
	Mat mat_resutl = CBusin_Opencv_Transform_Tool::instance().image_rotate(src_mat, center_point, 30, 1);
	cv::imshow("result:", mat_resutl);
	return 0;
}

int test_roate_without_lost(const cv::Mat& src_mat)
{
	cv::Point center_point;
	center_point.x = src_mat.cols/2 + 0.5;
	center_point.y = src_mat.rows/2 + 0.5;
	Mat mat_resutl = CBusin_Opencv_Transform_Tool::instance().rotate_image__and_shift(src_mat, center_point
		, 30, src_mat.cols * 0.2, src_mat.rows * 0.2);
	cv::imshow("result:", mat_resutl);
	return 0;
}
int test_flip(const Mat& src_mat)
{
	cv::flip(src_mat, src_mat, 0); //1:水平方向；0：竖直方向；-1：水平和竖直方向
	cv::imshow("result:", src_mat);
	return 0;
}
template<typename T>
int calc_consine(const vector<T> & vecA, const vector<T>& vecB, double& dCosine)
{
	if (vecA.size() != vecB.size())
	{
		std::cout << __FUNCTION__ << " | erro, line:" << __LINE__ << endl;
		return -1;
	}
	double vec_product = 0;//向量乘积
	double vecA_sum = 0;
	double vecB_sum = 0;
	for (int i= 0; i != vecA.size(); ++i)
	{
		vec_product += vecA[i] * vecB[i];
		vecA_sum += vecA[i] * vecA[i];
		vecB_sum += vecB[i] * vecB[i];
	}
	vecA_sum = std::sqrt(vecA_sum);
	vecB_sum = std::sqrt(vecB_sum);
	if (fabs(vecA_sum)  < 1e-6 || fabs(vecB_sum) < 1e-6 )
	{//其中有一个为0
		dCosine = 0;
		return 0;
	}
	double mold_product = vecA_sum * vecB_sum;
	if (fabs(mold_product) < 1e-6)
	{//模的积为0
		dCosine = 0;
	}
	else
	{
		dCosine =  vec_product / mold_product;
	}
	return 0;
}

template<class T>
bool is_Rect(const vector<T>& vec_points)
{
	double dThreshold = 0.1;
	T P0P1 = vec_points[1] - vec_points[0];
	T P1P2 = vec_points[2] - vec_points[1];
	//如果P0P1垂直于P1P2,则余弦值很接近0
	typename vector<float> vecA, vecB;
	vecA.push_back(P0P1.x);
	vecA.push_back(P0P1.y);
	vecB.push_back(P1P2.x);
	vecB.push_back(P1P2.y);
	double dCosine = 0;
	calc_consine<float>(vecA, vecB, dCosine);
	if (fabs(dCosine) > dThreshold)
	{
		return false;
	}
	vecA.clear();
	vecB.clear();
	T P2P3 = vec_points[3] - vec_points[2];
	T P3P0 = vec_points[0] - vec_points[3];
	
	vecA.push_back(P2P3.x);
	vecA.push_back(P2P3.y);
	vecB.push_back(P3P0.x);
	vecB.push_back(P3P0.y);
	dCosine = 0;
	calc_consine<float>(vecA, vecB, dCosine);
	if (fabs(dCosine) > dThreshold)
	{
		return false;
	}
	return true;
}
template<class T>
int sort_and_verify_rect_points(T* pPoints_arr, size_t nNum)
{
	vector<T> vec_points;
	for (int i = 0; i != nNum; ++i)
	{
		vec_points.push_back(pPoints_arr[i]);
	}
	//对顶点进行排序，判定是否能够组成矩形
	for (size_t idx_A = 0; idx_A != vec_points.size(); ++idx_A)
	{
		vector<T> temp_first_point_vec = vec_points;
		T point_A_local = temp_first_point_vec[idx_A];
		vector<T>::iterator iter_A = find(temp_first_point_vec.begin(), temp_first_point_vec.end(), point_A_local);
		temp_first_point_vec.erase(iter_A);
		for (size_t idx_B = 0; idx_B != temp_first_point_vec.size(); ++idx_B)
		{
			vector<T> temp_second_point_vec = temp_first_point_vec;
			T point_B_local = temp_second_point_vec[idx_B];
			vector<T>::iterator iter_B = find(temp_second_point_vec.begin(), temp_second_point_vec.end(), point_B_local);
			temp_second_point_vec.erase(iter_B);
			for (size_t idx_C = 0; idx_C != temp_second_point_vec.size(); ++idx_C)
			{
				vector<T> temp_third_point_vec = temp_second_point_vec;
				T point_C_local = temp_third_point_vec[idx_C];
				vector<T>::iterator iter_C = find(temp_third_point_vec.begin(), temp_third_point_vec.end(), point_C_local);
				temp_third_point_vec.erase(iter_C);
				
				for (size_t idx_D = 0; idx_D != temp_third_point_vec.size(); ++idx_D)
				{
					T point_D_local = temp_third_point_vec[idx_D];
					vector<T> vec_tmp;
					vec_tmp.push_back(point_A_local);
					vec_tmp.push_back(point_B_local);
					vec_tmp.push_back(point_C_local);
					vec_tmp.push_back(point_D_local);
					std::cout << "point_A:" << point_A_local << ", point_B" <<point_B_local 
						<< ", point_C" <<point_C_local << ",point_D" << point_D_local<< endl;
					bool is_rect = is_Rect<T>(vec_tmp);
					if (is_rect)
					{
						for (int i = 0; i != nNum; ++i)
						{
							pPoints_arr[i] = vec_tmp[i];
						}
						return 0;
					}
				}
			}
		}
	}
	return -1;
}
int get_my_RotatedRect()
{
	Point2f points_arr[4] = {Point2f(-77.611, -31.049), Point2f(-72.599, -20.323), Point2f(-58.841, -27.265), Point2f(-64.354, -37.991)};
	const Point2f& _point1 = points_arr[0];
	const Point2f& _point2 = points_arr[1];
	const Point2f& _point3 = points_arr[2];
	Point2f _center = 0.5f * (_point1 + _point3);
	Vec2f vecs[2];
	vecs[0] = Vec2f(_point1 - _point2);
	vecs[1] = Vec2f(_point2 - _point3);
	// check that given sides are perpendicular
//	CV_Assert( abs(vecs[0].dot(vecs[1])) / (norm(vecs[0]) * norm(vecs[1])) <= FLT_EPSILON );

	// wd_i stores which vector (0,1) or (1,2) will make the width
	// One of them will definitely have slope within -1 to 1
	int wd_i = 0;
	if( abs(vecs[1][1]) < abs(vecs[1][0]) ) wd_i = 1;
	int ht_i = (wd_i + 1) % 2;

	float _angle = atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (float) CV_PI;
	float _width = (float) norm(vecs[wd_i]);
	float _height = (float) norm(vecs[ht_i]);
	RotatedRect src_dst_rRect;
	src_dst_rRect.center = _center;
	src_dst_rRect.size = Size2f(_width, _height);
	src_dst_rRect.angle = _angle;
	return 0;
}
void test_opencv()
{
	Point2f points_arr[4] = {Point2f(-77.611, -31.049), Point2f(-72.599, -20.323), Point2f(-58.841, -27.265), Point2f(-64.354, -37.991)};
	RotatedRect src_dst_rRect(points_arr[0], points_arr[1], points_arr[2]);
// 	Point points_dst_rect_list[4] = {Point(891, 622), Point(991, 407), Point(1267, 546), Point(1157, 761)};
// 	const Point* pDst_rect_points =  points_dst_rect_list;
// 	RotatedRect src_dst_rRect(Point2f(pDst_rect_points[0].x, pDst_rect_points[0].y)
// 		, Point2f(pDst_rect_points[1].x, pDst_rect_points[1].y)
// 		, Point2f(pDst_rect_points[2].x, pDst_rect_points[2].y));
}
int test_sort_rect()
{
	Point2f points_arr[4] = {Point2f(-77.611, -31.049), Point2f(-72.599, -20.323), Point2f(-58.841, -27.265), Point2f(-64.354, -37.991)};
//	Point2f points_dst_rect_list[4] = {Point2f(891, 622), Point2f(991, 407), Point2f(1267, 546), Point2f(1157, 761)};
	int ret =  sort_and_verify_rect_points<Point2f>(points_arr, 4);
	if (ret)
	{
		std::cout << __FUNCTION__ <<" | line:" <<__LINE__ << ", ret:" << ret << std::endl;
	}
	return ret;
}
int test_RotatedRect()
{
	Mat image(200, 200, CV_8UC3, Scalar(0));
	RotatedRect rRect = RotatedRect(Point2f(100,100), Size2f(50, 100), -60);

	Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));

	Rect brect = rRect.boundingRect();
	rectangle(image, brect, Scalar(255,0,0));
	image.at<Vec3b>(vertices[0]) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x +1, vertices[0].y + 1)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x +1, vertices[0].y - 1)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x - 1, vertices[0].y + 1)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x - 1, vertices[0].y - 1)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x , vertices[0].y + 1)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x , vertices[0].y - 1)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x - 1 , vertices[0].y)) = Vec3b(0,0,255);
	image.at<Vec3b>(Point(vertices[0].x + 1 , vertices[0].y)) = Vec3b(0,0,255);
	imshow("rectangles", image);
	RotatedRect tmp_rRect(vertices[0], vertices[1], vertices[2]);
	waitKey(0);
	return 0;
}
int test_gray_mat()
{
	const string str_src_img_path = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_src/guxiaowei.jpg";
	const string str_dst_img_path = "F:/GitHub/OpenCVTest/trunk/OpencvTest/images_result/guxiaowei_gray.jpg";
	cv::Mat mat_src = imread(str_src_img_path, IMREAD_GRAYSCALE);
	if (mat_src.empty())
	{
		printf("%s | fail to read img:%s\n", __FUNCTION__, str_src_img_path.c_str());
		return -1;
	}

	double dAlpha = 2;
	int nBeta = 0;
	cout <<" Basic Linear Transforms "<<endl;
	cout <<"-------------------------"<<endl;
	cout <<" *Enter the alpha value [1.0-3.0]: ";
	cin >> dAlpha;
	cout <<" *Enter the beta value [0-100]: "; 
	cin >> nBeta;
	for (int nRow = 0; nRow != mat_src.rows; ++nRow)
	{
		for (int nCol = 0; nCol != mat_src.cols; ++nCol)
		{
			mat_src.at<uchar>(nRow, nCol) = saturate_cast<uchar>( dAlpha * (mat_src.at<uchar>(nRow, nCol)) + nBeta );
		}
	}
	imwrite(str_dst_img_path, mat_src);

	return 0;
}
//将浮点数转换为字符串时，不使用科学记数法
void test_nonKXJSF()
{
	ostringstream oss;
	double fVal = 0.00001;
	oss << setiosflags(ios::fixed) << fVal;
	std::cout << oss.str() << endl;
}
void test_create_mat()
{
	//从一个已知数组中构造矩阵
	double arr_f1[3][3] = { 0.1, 0.1, 0.1
	                     , 0.2, 0.2, 0.2
	                     , 0.3, 0.3, 0.3};
	Mat mat_f1(3, 3, CV_32FC1, arr_f1);
	std::cout << "mat_F1:" << mat_f1 << std::endl;
	arr_f1[2][2] = 15;
	//此时mat_f1已经被修改
	std::cout << "mat_F1 after modified:" << mat_f1 << std::endl;
	Rect my_rect(0, 0, 2, 2);
	Mat mat_ri(mat_f1, my_rect);
}
void test_create_black_background_img()
{
	string str_img_path = "./images_for_MER/2_rotated.jpg";
	Mat mat_src_gray = imread(str_img_path, COLOR_BGR2GRAY);
	//求二值灰度矩阵(黑色为底)
	Mat mat_src_binary_gray;
	threshold(mat_src_gray, mat_src_binary_gray, 100, 255,THRESH_BINARY_INV);
//	cvtColor(mat_src_binary_gray, mat_src_binary_gray, COLOR_GRAY2BGR);
	imwrite(str_img_path+"_binary_gray.jpg", mat_src_binary_gray);
}
int main(int argc, char** argv)
{
	try
	{
//		CBusin_OpenCV_Vector_Tool::instance().test_find_PointB();
//		CBusin_OpenCV_Common_Tool::instance().test_detect_circles("F:/project/CoreRepo/Development/Source/trunk/comnon_sdk/bin/mtrec_scp/mt_scylla/ees_detect_and_transform/circle/0911/before.jpg");
//		CBusin_OpenCV_File_Tool::instance().test(argc, argv);
//		CBusin_OpenCV_Common_Tool::instance().test_transform();
//		CBusin_OpenCV_Common_Tool::instance().test_change_contrast_and_brightness("F:/project/CoreRepo/Development/Source/trunk/comnon_sdk/bin/mtrec_scp/mt_scylla/ees_detect_and_transform/circle/0911/before.jpg");
//		CBusin_OpenCV_Common_Tool::instance().test_wenzi_G_code("./wenzi.jpg");
//		CBusin_OpenCV_Common_Tool::instance().test_wenzi_G_code("./weinzi_shinked_result.jpg");
//		CBusin_OpenCV_Common_Tool::instance().test_shrink_mat();
//		test_RotatedRect();
//		test_sort_rect();
//		test_opencv();
//		get_my_RotatedRect();
//		CBusin_OpenCV_Common_Tool::instance().test_shrink("F:/project/CoreRepo/Development/Source/trunk/comnon_sdk/bin/mtrec_scp/mt_scylla/ee_handwriting_test/wenzi_luming_right.jpg");
// 		CBusin_OpenCV_Common_Tool::instance().test_dilate(
// 			"F:/project/CoreRepo/Development/Source/trunk/comnon_sdk/bin/mtrec_scp/mt_scylla/ee_handwriting_test/wenzi_luming_right_with_line.jpg");
		//	    CBusin_OpenCV_Common_Tool::instance().test_get_binary_gray_mat("F:\\project\\Programming_Test\\OpencvTest\\OpencvTest\\images_src");
//		test_gray_mat();
//		CBusin_OpenCV_Contour_Tool::instance().test("F:/GitHub/OpenCVTest/trunk/OpencvTest/images_src/xiaba_jll.jpg");
//	    CInscribed_Polygon_Tool::instace().test_max_inscribed_rect_using_traversing_for_rotated2();
//		test_create_black_background_img();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_sketch();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_GaoFanChaBaoLiu();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_photocopy();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_Laplacian_sketch();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_Sobel_sketch();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_differenceOfGaussian();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_difference_IPLB();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_difference_Edge_Detect();
//		CBusin_OpenCV_Filter_Tool_Inst::instance().test_photocopy_GIMP();
//		CBusin_OpenCV_Connected_Graph::instance().test_connected_graph_img();
//		CBusin_OpenCV_Connected_Graph::instance().test_connected_graph_arr();
		CBusin_OpenCV_Common_Tool::instance().test_Catmull_Rom();
		CBusin_OpenCV_Filter_Tool_Inst::instance().test_draw_chin();
	}
	catch (std::exception& e)
	{
		std::cout << "fun:" << __FUNCTION__ << ", error reason:" << e.what() << std::endl;
	}
	return 0;
}
