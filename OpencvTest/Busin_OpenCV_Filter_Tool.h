#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
class CBusin_OpenCV_Filter_Tool_Inst
{
public:
	static CBusin_OpenCV_Filter_Tool_Inst& instance();
	int test_sketch();
	int test_Laplacian_sketch();

	int test_GaoFanChaBaoLiu();
	int test_photocopy();
protected:
private:
	static CBusin_OpenCV_Filter_Tool_Inst ms_inst;
};