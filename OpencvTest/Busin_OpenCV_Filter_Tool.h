#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/mat.hpp>
using namespace cv;
enum EMBOSS_DIRECTION
{
	N,
	NE,
	E,
	SE,
	S,
	SW,
	W,
	NW
};

class CBusin_OpenCV_Filter_Tool_Inst
{
public:
	static CBusin_OpenCV_Filter_Tool_Inst& instance();
	int test_sketch();
	int test_Laplacian_sketch();
	int test_Sobel_sketch();
	int test_Scharr_sketch()
	{
		//Scharr算子
		return 0;
	}
	int test_differenceOfGaussian();
	int test_difference_IPLB();
	int test_GaoFanChaBaoLiu();
	int test_photocopy();
	int test_difference_Edge_Detect();
protected:
	//基于拉普拉斯锐化算法的素描算法
	void Sketch(const Mat& img, Mat& dst);
	//基于八方向浮雕算法的素描算法
	void DiamondEmboss(const Mat& img, Mat& dst, EMBOSS_DIRECTION Ed /* = SE */,int offset /* = 127 */);
	int getPixel(const Mat& mat_img, int y, int x, int channel);
	//高斯差分
	int differenceOfGaussian(const Mat& mat_src, Mat& mat_dst);
	void difference_Edge_Detect( const Mat& mat_src_gray, cv::Mat& mat_dst_gray, const cv::Rect& rect);
private:
	static CBusin_OpenCV_Filter_Tool_Inst ms_inst;
};