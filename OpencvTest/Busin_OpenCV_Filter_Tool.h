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
typedef struct
{
	double  mask_radius; //蒙板半径，即强度比较的像素邻域半径，在该上计算平均强度，然后与邻域中的每个像素进行比较
	                     //，以决定是否将其变暗为黑色.其值越大，黑色区域越密集，细节越少
	double  sharpness;//锐度
	double  threshold;//相对强度差异阈值，此值决定是否暗化
	double  pct_black;//黑色像素百分比，此值决定着变暗为黑色时的黑色总量。值越小，使得从白色区域到合适边框线
	                  //的混合更平滑，调色区域更稀疏并且不太明显
	double  pct_white;//白色像素百分比
} PhotocopyVals;

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
	int test_photocopy_myself();
	int test_photocopy_GIMP();
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
	//photocopy（legacy）
	void photocopy_gimp(const Mat& mat_src_gray, size_t nMask_radius, double dThreshold, Mat& mat_gray_result);

	//自己按算法公式来写的
	int photocopy_myself(const Mat& mat_src_gray, size_t nMask_radius, double dThreshold, Mat& mat_gray_result);
private:
	//影印滤镜相关
	//https://www.spinics.net/lists/gimpdev/msg26041.html
	/*
	*  Gaussian blur helper functions
	*/
	static void transfer_pixels(double *pdScr1, double *pdSrc2, uchar  *pcDest, int nJump, int nWidth);

	static void find_constants(double n_p[], double n_m[],double d_p[], double d_m[], double bd_p[], double bd_m[], double std_dev);
	
	/************************************
	* Method:    compute_ramp
	* Brief:  根据两组字节流，计算它们之间的斜面
	* Access:    private static 
	* Returns:   double
	* Qualifier:
	*Parameter: uchar * dest1 -[in] 图片1字节流 
	*Parameter: uchar * dest2 -[in]  图片2字节流
	*Parameter: int length -[in]  字节流长度
	*Parameter: double pct_black -[in] 黑色像素点百分比  
	*Parameter: int under_threshold -[in] 低阈值 
	************************************/
	static double compute_ramp(uchar  *dest1, uchar  *dest2, int length, double pct_black, int under_threshold);
	static CBusin_OpenCV_Filter_Tool_Inst ms_inst;
};