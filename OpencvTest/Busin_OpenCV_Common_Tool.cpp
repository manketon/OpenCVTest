/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: Busin_OpenCV_Common_Tool.cpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:	minglu2
* @version: 1.0
* @date: 2019/06/11
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2019/06/11	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#include "Busin_OpenCV_Common_Tool.h"



int CBusin_OpenCV_Common_Tool::get_my_RotatedRect(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3, RotatedRect& rRect)
{
	Point2f _center = 0.5f * (_point1 + _point3);
	Vec2f vecs[2];
	vecs[0] = Vec2f(_point1 - _point2);
	vecs[1] = Vec2f(_point2 - _point3);
	// wd_i stores which vector (0,1) or (1,2) will make the width
	// One of them will definitely have slope within -1 to 1
	int wd_i = 0;
	if( abs(vecs[1][1]) < abs(vecs[1][0]) ) wd_i = 1;
	int ht_i = (wd_i + 1) % 2;

	float _angle = atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (float) CV_PI;
	float _width = (float) norm(vecs[wd_i]);
	float _height = (float) norm(vecs[ht_i]);

	rRect.center = _center;
	rRect.size = Size2f(_width, _height);
	rRect.angle = _angle;
	return 0;
}

int CBusin_OpenCV_Common_Tool::change_contrast_and_brightness(const Mat& mat_src_bgr, double udContrast, int nBrightness, Mat& mat_dst_image)
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
// 	for( int y = 0; y < mat_src_bgr.rows; y++ )
// 	{
// 		for( int x = 0; x < mat_src_bgr.cols; x++ )
// 		{
// 			for( int c = 0; c < 3; c++ )
// 			{
// 				mat_dst_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( udContrast*( mat_src_bgr.at<Vec3b>(y,x)[c] ) + nBrightness );
// 			}
// 		}
// 	}
	//下面逻辑运行速度快
	int nr = mat_src_bgr.rows;
	int nc = mat_src_bgr.cols * mat_src_bgr.channels() ;
	if (mat_src_bgr.isContinuous() && mat_dst_image.isContinuous())
	{
		nr = 1;
		nc = nc * mat_src_bgr.rows;
	}
	for ( int i = 0; i < nr; ++i)
	{
		const uchar* pSrc_data = mat_src_bgr.ptr<uchar>(i);
		uchar* pDst_data = mat_dst_image.ptr<uchar>(i);
		for (int j = 0; j < nc; ++j)
		{
			pDst_data[j] = saturate_cast<uchar>(udContrast * pSrc_data[j] + nBrightness);
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
