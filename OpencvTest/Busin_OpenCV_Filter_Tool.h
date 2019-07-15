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
	double  mask_radius; //�ɰ�뾶����ǿ�ȱȽϵ���������뾶���ڸ��ϼ���ƽ��ǿ�ȣ�Ȼ���������е�ÿ�����ؽ��бȽ�
	                     //���Ծ����Ƿ���䰵Ϊ��ɫ.��ֵԽ�󣬺�ɫ����Խ�ܼ���ϸ��Խ��
	double  sharpness;//���
	double  threshold;//���ǿ�Ȳ�����ֵ����ֵ�����Ƿ񰵻�
	double  pct_black;//��ɫ���ذٷֱȣ���ֵ�����ű䰵Ϊ��ɫʱ�ĺ�ɫ������ֵԽС��ʹ�ôӰ�ɫ���򵽺��ʱ߿���
	                  //�Ļ�ϸ�ƽ������ɫ�����ϡ�貢�Ҳ�̫����
	double  pct_white;//��ɫ���ذٷֱ�
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
		//Scharr����
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
	//����������˹���㷨�������㷨
	void Sketch(const Mat& img, Mat& dst);
	//���ڰ˷��򸡵��㷨�������㷨
	void DiamondEmboss(const Mat& img, Mat& dst, EMBOSS_DIRECTION Ed /* = SE */,int offset /* = 127 */);
	int getPixel(const Mat& mat_img, int y, int x, int channel);
	//��˹���
	int differenceOfGaussian(const Mat& mat_src, Mat& mat_dst);
	void difference_Edge_Detect( const Mat& mat_src_gray, cv::Mat& mat_dst_gray, const cv::Rect& rect);
	//photocopy��legacy��
	void photocopy_gimp(const Mat& mat_src_gray, size_t nMask_radius, double dThreshold, Mat& mat_gray_result);

	//�Լ����㷨��ʽ��д��
	int photocopy_myself(const Mat& mat_src_gray, size_t nMask_radius, double dThreshold, Mat& mat_gray_result);
private:
	//Ӱӡ�˾����
	//https://www.spinics.net/lists/gimpdev/msg26041.html
	/*
	*  Gaussian blur helper functions
	*/
	static void transfer_pixels(double *pdScr1, double *pdSrc2, uchar  *pcDest, int nJump, int nWidth);

	static void find_constants(double n_p[], double n_m[],double d_p[], double d_m[], double bd_p[], double bd_m[], double std_dev);
	
	/************************************
	* Method:    compute_ramp
	* Brief:  ���������ֽ�������������֮���б��
	* Access:    private static 
	* Returns:   double
	* Qualifier:
	*Parameter: uchar * dest1 -[in] ͼƬ1�ֽ��� 
	*Parameter: uchar * dest2 -[in]  ͼƬ2�ֽ���
	*Parameter: int length -[in]  �ֽ�������
	*Parameter: double pct_black -[in] ��ɫ���ص�ٷֱ�  
	*Parameter: int under_threshold -[in] ����ֵ 
	************************************/
	static double compute_ramp(uchar  *dest1, uchar  *dest2, int length, double pct_black, int under_threshold);
	static CBusin_OpenCV_Filter_Tool_Inst ms_inst;
};