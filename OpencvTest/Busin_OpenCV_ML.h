/******************************************************************
* Copyright (c) 2004-2019,TaiHo Inc.
* All rights reserved.
* @file: Busin_OpenCV_ML.h
* @brief: 使用OpenCV的机器学习模块来实现相关算法
* @author:  minglu2
* @version: 1.0
* @date: 2020/04/29
*
* @see
*
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本    <th>日期        <th>作者    <th>备注 </tr>
*  <tr> <td>1.0     <td>2020/04/29  <td>minglu  <td>Create head file </tr>
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
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>

//宏定义

//类型定义
class CBusin_OpenCV_ML_Tool
{
public:
	static CBusin_OpenCV_ML_Tool& instance();
	int test();
	
protected:
	//在数据集letter-recognition.data上，测试OpenCV的随机森林算法
	int test_random_forests();
	/************************************
	// Method:    read_num_class_data
	// Brief:     从文件中读取data和responses 
	// Access:    protected 
	// Returns:   int MSP_SUCCESS:函数调用成功
	// Parameter: const string & filename -[in/out]
	// Parameter: int nFeature_vec_elements_num -[in] 特征向量元素数目
	// Parameter: Mat * _data -[out] 数据集
	// Parameter: Mat * _responses -[out] 标签集 
	*************************************/
	int read_num_class_data( const std::string& filename, int nFeature_vec_elements_num, cv::Mat* _data, cv::Mat* _responses );
	//准备训练数据
	cv::Ptr<cv::ml::TrainData> prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int nTrain_samples_num);

	//设置迭代条件
	cv::TermCriteria set_iteration_conditions(int iters, double eps);

	//分类预测
	void test_and_save_classifier(const cv::Ptr<cv::ml::StatModel>& model,    const cv::Mat& data, const cv::Mat& responses,
		int nTrain_samples_num, int rdelta);

	//随机树分类
	bool build_rtrees_classifier(const std::string& data_filename);

	//adaboost分类
	bool build_boost_classifier(const std::string& data_filename);

	//多层感知机分类（ANN）
	bool build_mlp_classifier(const std::string& data_filename);

	//K最近邻分类
	bool build_knearest_classifier(const std::string& data_filename, int K);

	//贝叶斯分类
	bool build_nbayes_classifier(const std::string& data_filename);


	//svm分类
	bool build_svm_classifier(const std::string& data_filename);

	int test_SVM_KNN_RTree();
	//颜色识别相关
	int test_SVM_color_recognition();
	//测试神经网络
	int test_ANN_1();
	int test_ANN_2();

	//从指定目录中加载数据
	int get_labels_and_class_data(const std::string& str_src_imgs_dir,  const std::string& str_class_prefix
		, cv::Mat& mat_trains_dataes, cv::Mat& mat_trains_lables);
	//从路径中获取类别号,路径格式为xxx/类别前缀1/yyy.bmp,如xxx/class_1/yyy
	int get_class_num_from_path(const std::string& str_file_path, const std::string& str_class_prefix
		, int& nLabel);
	bool is_background(const cv::Vec3b& vec3b_pixel)
	{
		if (vec3b_pixel[0] > 90 && vec3b_pixel[0] > vec3b_pixel[1] +  20 
			&& vec3b_pixel[0] > vec3b_pixel[2] + 20)
		{
			return true;
		}
		return false;
	}
	int create_gray_with_model(const cv::Ptr<cv::ml::StatModel>& model);

private:
};
//函数原型定义