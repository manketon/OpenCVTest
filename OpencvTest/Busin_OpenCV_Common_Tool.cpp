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
