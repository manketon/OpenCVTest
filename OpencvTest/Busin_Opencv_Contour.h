#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

class CBusin_OpenCV_Contour_Tool
{
public:
	static CBusin_OpenCV_Contour_Tool& instance();
	/** @function main */
	int test(const string& str_img_file_path);

protected:
private:
};
