#include "Busin_OpenCV_ML.h"
#include <iostream>
#include "File_System_Tool.h"
#include <boost/lexical_cast.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include<boost/date_time/posix_time/posix_time.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

CBusin_OpenCV_ML_Tool& CBusin_OpenCV_ML_Tool::instance()
{
	static CBusin_OpenCV_ML_Tool obj;
	return obj;
}

int CBusin_OpenCV_ML_Tool::test()
{
	while (true)
	{
		std::cout << "1、测试随机森林算法" << std::endl;
		std::cout << "2、在OpenCV的OCR数据集上同时测试SVM、贝叶斯、随机森林等分类器" << std::endl;
		std::cout << "3、SVM来识别颜色" << std::endl;
		std::cout << "4、神经网络实例1" << std::endl;
		std::cout << "5、神经网络实例2" << std::endl;
		std::cout << "0、退出机器学习模块" << std::endl;
		std::cout << "请输入您的选择:";
		int nChoise = 0;
		std::cin >> nChoise;
		switch (nChoise)
		{
		case 0:
			cout << __FUNCTION__ << " | 退出成功" << endl;
			return 0;
		case 1:
			test_random_forests();
			break;
		case 2:
			test_SVM_KNN_RTree();
			break;
		case 3:
			test_SVM_color_recognition();
			break;
		case 4:
			test_ANN_1();
			break;
		case 5:
			test_ANN_2();
			break;
		default:
			break;
		}
	}
	return 0;
}

int CBusin_OpenCV_ML_Tool::test_random_forests()
{
	string data_filename = "E:/opencv_3.3.1_install/opencv/sources/samples/data/letter-recognition.data";
	Mat data; //特征向量数据集
	Mat responses; //标签数据集
	//读取data和responses
	read_num_class_data( data_filename, 16, &data, &responses );

	int nsamples_all = data.rows;  //样本总数
	int ntrain_samples = (int)(nsamples_all*0.8);  //训练样本个数
	cout << "Training the classifier ...\n"<<endl;
	Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
	int nvars = data.cols; //一个特征向量中的特征数目，即特征维数
	Mat var_type( nvars + 1, 1, CV_8U );
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;
	//训练数据
	Ptr<TrainData> tdata = TrainData::create(data, ROW_SAMPLE, responses, noArray(), noArray()/*sample_idx*/, noArray(), noArray()/*var_type*/);
	// 创建分类器
	Ptr<RTrees> model = RTrees::create();
	//树的最大可能深度
	model->setMaxDepth(10);
	//节点最小样本数量
	model->setMinSampleCount(10);
	//回归树的终止标准
	model->setRegressionAccuracy(0);
	//是否建立替代分裂点
	model->setUseSurrogates(false);
	//最大聚类簇数
	model->setMaxCategories(15);
	//先验类概率数组
	model->setPriors(Mat());
	//计算的变量重要性
	model->setCalculateVarImportance(true);
	//树节点随机选择的特征子集的大小
	model->setActiveVarCount(4);
	//终止标准
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + (0.01f > 0 ? TermCriteria::EPS : 0), 100, 0.01f));
	//训练模型
	model->train(tdata);
	//保存训练完成的模型
	//model->save("filename_to_save.xml");

	double train_hr = 0, test_hr = 0;
	// 计算训练和测试数据的预测误差
	for( int i = 0; i < nsamples_all; i++ )
	{
		Mat sample = data.row(i);
		float r = model->predict( sample );
		//判断预测是否正确（绝对值小于最小值FLT_EPSILON）
		r = std::abs(r - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		//计数
		if( i < ntrain_samples )
			train_hr += r;
		else
			test_hr += r;
	}
	//训练数据的预测误差
	test_hr /= nsamples_all - ntrain_samples;
	//测试数据的预测误差
	train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;
	printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n", train_hr*100., test_hr*100. );
	//随机森林中的树个数
	cout << "Number of trees: " << model->getRoots().size() << endl;
	// 特征重要性
	Mat mat_features_importance = model->getVarImportance();
	if( !mat_features_importance.empty() )
	{
		double rt_imp_sum = sum( mat_features_importance )[0];
		printf("feature#\timportance (in %%):\n");
		int i, n = (int)mat_features_importance.total();
		for( i = 0; i < n; i++ )
			printf( "%-2d\t%-4.1f\n", i, 100.f*mat_features_importance.at<float>(i)/rt_imp_sum);
	}
	return 0;
}

int CBusin_OpenCV_ML_Tool::read_num_class_data(const string& filename, int nFeature_vec_elements_num, cv::Mat* _data, cv::Mat* _responses)
{
	const int M = 1024;
	char buf[M+2] = {0};
	Mat el_ptr(1, nFeature_vec_elements_num, CV_32F); 
	vector<int> responses;
	_data->release();
	_responses->release();
	//f指向存储数据的地址
	FILE* f = fopen( filename.c_str(), "rt" );
	if( !f )
	{
		cout << "Could not read the database " << filename << endl;
		return -1;
	}

	for(;;)
	{
		char* ptr;
		int i;
		//fgets从文件中读取一行数据存入缓冲区
		//strchr查找字符串buf中首次出现，的位置
		if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
			break;
		responses.push_back((int)buf[0]);
		ptr = buf + 2;
		for( i = 0; i < nFeature_vec_elements_num; i++ )
		{
			int n = 0;
			//读取格式化的字符串中的数据
			sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
			ptr += n + 1;
		}
		if( i < nFeature_vec_elements_num )
			break;
		_data->push_back(el_ptr);
	}
	fclose(f);

	Mat(responses).copyTo(*_responses);
	cout << "The database " << filename << " is loaded.\n";
	return 0;
}

cv::Ptr<cv::ml::TrainData> CBusin_OpenCV_ML_Tool::prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int nTrain_samples_num)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
//	Mat train_samples = sample_idx.colRange(0, nTrain_samples_num);
//	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols; //每个特征向量中的特征数目
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

cv::TermCriteria CBusin_OpenCV_ML_Tool::set_iteration_conditions(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

void CBusin_OpenCV_ML_Tool::test_and_save_classifier(const cv::Ptr<cv::ml::StatModel>& model, const cv::Mat& data, const cv::Mat& responses, int nTrain_samples_num, int rdelta)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);
		int nLabel_src = responses.at<int>(i);
		float r = model->predict(sample);
		int nSuccess = std::abs(r + rdelta - nLabel_src) <= FLT_EPSILON ? 1 : 0;
//		std::cout << __FUNCTION__ << " | predict label:" << r << ", src label:" << nLabel_src << ", sample:" << sample << endl;
		if (i < nTrain_samples_num)
			train_hr += nSuccess;
		else
			test_hr += nSuccess;
	}

	test_hr /= nsamples_all - nTrain_samples_num;
	train_hr = nTrain_samples_num > 0 ? train_hr / nTrain_samples_num : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);
}

bool CBusin_OpenCV_ML_Tool::build_rtrees_classifier(const std::string& data_filename)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 16, &data, &responses);

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<RTrees> model;
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = RTrees::create();
	model->setMaxDepth(10);
	model->setMinSampleCount(10);
	model->setRegressionAccuracy(0);
	model->setUseSurrogates(false);
	model->setMaxCategories(15);
	model->setPriors(Mat());
	model->setCalculateVarImportance(true);
	model->setActiveVarCount(4);
	model->setTermCriteria(set_iteration_conditions(100, 0.01f));
	model->train(tdata);
	test_and_save_classifier(model, data, responses, ntrain_samples, 0);
	cout << "Number of trees: " << model->getRoots().size() << endl;

	// Print variable importance
	Mat var_importance = model->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (in %%):\n");
		int i, n = (int)var_importance.total();
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}

	return true;
}

bool CBusin_OpenCV_ML_Tool::build_boost_classifier(const std::string& data_filename)
{
	const int class_count = 26;
	Mat data;
	Mat responses;
	Mat weak_responses;

	read_num_class_data(data_filename, 16, &data, &responses);
	int i, j, k;
	Ptr<Boost> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.5);
	int var_count = data.cols;

	Mat new_data(ntrain_samples*class_count, var_count + 1, CV_32F);
	Mat new_responses(ntrain_samples*class_count, 1, CV_32S);

	for (i = 0; i < ntrain_samples; i++)
	{
		const float* data_row = data.ptr<float>(i);
		for (j = 0; j < class_count; j++)
		{
			float* new_data_row = (float*)new_data.ptr<float>(i*class_count + j);
			memcpy(new_data_row, data_row, var_count*sizeof(data_row[0]));
			new_data_row[var_count] = (float)j;
			new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j + 'A';
		}
	}

	Mat var_type(1, var_count + 2, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count + 1) = VAR_CATEGORICAL;

	Ptr<TrainData> tdata = TrainData::create(new_data, ROW_SAMPLE, new_responses,
		noArray(), noArray(), noArray(), var_type);
	vector<double> priors(2);
	priors[0] = 1;
	priors[1] = 26;

	model = Boost::create();
	model->setBoostType(Boost::GENTLE);
	model->setWeakCount(100);
	model->setWeightTrimRate(0.95);
	model->setMaxDepth(5);
	model->setUseSurrogates(false);
	model->setPriors(Mat(priors));
	model->train(tdata);
	Mat temp_sample(1, var_count + 1, CV_32F);
	float* tptr = temp_sample.ptr<float>();

	// compute prediction error on train and test data
	double train_hr = 0, test_hr = 0;
	for (i = 0; i < nsamples_all; i++)
	{
		int best_class = 0;
		double max_sum = -DBL_MAX;
		const float* ptr = data.ptr<float>(i);
		for (k = 0; k < var_count; k++)
			tptr[k] = ptr[k];

		for (j = 0; j < class_count; j++)
		{
			tptr[var_count] = (float)j;
			float s = model->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);
			if (max_sum < s)
			{
				max_sum = s;
				best_class = j + 'A';
			}
		}

		double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ? 1 : 0;
		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;
	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	cout << "Number of trees: " << model->getRoots().size() << endl;
	return true;
}

bool CBusin_OpenCV_ML_Tool::build_mlp_classifier(const std::string& data_filename)
{
	const int class_count = 26;
	Mat data;
	Mat responses;

	read_num_class_data(data_filename, 16, &data, &responses);
	Ptr<ANN_MLP> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);
	Mat train_data = data.rowRange(0, ntrain_samples);
	Mat train_responses = Mat::zeros(ntrain_samples, class_count, CV_32F);

	// 1. unroll the responses
	cout << "Unrolling the responses...\n";
	for (int i = 0; i < ntrain_samples; i++)
	{
		int cls_label = responses.at<int>(i) -'A';
		train_responses.at<float>(i, cls_label) = 1.f;
	}

	// 2. train classifier
	int layer_sz[] = { data.cols, 100, 100, class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.001;
	int max_iter = 300;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(set_iteration_conditions(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
	return true;
}

bool CBusin_OpenCV_ML_Tool::build_knearest_classifier(const std::string& data_filename, int K)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 16, &data, &responses);
	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tdata);

	test_and_save_classifier(model, data, responses, ntrain_samples, 0);
	return true;
}

bool CBusin_OpenCV_ML_Tool::build_nbayes_classifier(const std::string& data_filename)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 16, &data, &responses);

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<NormalBayesClassifier> model;
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = NormalBayesClassifier::create();
	model->train(tdata);

	test_and_save_classifier(model, data, responses, ntrain_samples, 0);
	return true;
}

bool CBusin_OpenCV_ML_Tool::build_svm_classifier(const std::string& data_filename)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 16, &data, &responses);

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<SVM> model;
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = SVM::create();
	model->setType(SVM::C_SVC);
	model->setKernel(SVM::LINEAR);
	model->setC(1);
	model->train(tdata);

	test_and_save_classifier(model, data, responses, ntrain_samples, 0);
	return true;
}

int CBusin_OpenCV_ML_Tool::test_SVM_KNN_RTree()
{
	string data_filename = "E:/opencv_3.3.1_install/opencv/sources/samples/data/letter-recognition.data";  //字母数据

	cout << "svm分类：" << endl;
	build_svm_classifier(data_filename);

	cout << "贝叶斯分类：" << endl; 
	build_nbayes_classifier(data_filename);

	cout << "K最近邻分类：" << endl;
	build_knearest_classifier(data_filename,10);    

	cout << "随机森林分类：" << endl;
	build_rtrees_classifier(data_filename);

	cout << "adaboost分类：" << endl;
	build_boost_classifier(data_filename);

	//cout << "ANN（多层感知机)分类：" << endl;
	//build_mlp_classifier(data_filename);
	return 0;
}

int CBusin_OpenCV_ML_Tool::test_SVM_color_recognition()
{
	string str_src_imgs_dir;
	cout << "请输入训练数据父目录（子目录名为前缀_类别号）:";
	cin >> str_src_imgs_dir;

	string str_dst_dir;
	cout << "请输入模型的存储目录地址:";
	cin >> str_dst_dir;

	Mat mat_data;
	Mat mat_labels;
	string str_class_prefix("class_");
	//加载数据和标签
	get_labels_and_class_data(str_src_imgs_dir, str_class_prefix, mat_data, mat_labels);

	int nsamples_all = mat_data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);
	Ptr<TrainData> tdata = prepare_train_data(mat_data, mat_labels, ntrain_samples);
#if 1
	Ptr<SVM> model;

	model = SVM::create();
	model->setType(SVM::C_SVC);
	model->setKernel(SVM::RBF/*SVM::LINEAR*/);
	model->setC(1);
	model->train(tdata);
#else
	int K = 3;
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tdata);
#endif
	test_and_save_classifier(model, mat_data, mat_labels, ntrain_samples, 0);
	//获取时间戳
	string str_time = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
	string str_model_file_path = str_dst_dir + "/color_recognition_svm_" + str_time + ".pkl";
	//保存模型
 	model->save(str_model_file_path);
	//使用模型来预测，并生成灰度图
	int ret = create_gray_with_model(model);
	if (ret)
	{
		std::cout << __FUNCTION__ << " | fail to create gray with model, ret:" << ret << endl;
		return ret;
	}
	return 0;
}

int CBusin_OpenCV_ML_Tool::test_ANN_1()
{
	//create random training data
	Mat_<float> data(100, 100);
	cv::randn(data, Mat::zeros(1, 1, data.type()), Mat::ones(1, 1, data.type()));
	//half of the samples for each class
	Mat_<float> responses(data.rows, 2);
	for (int i = 0; i<data.rows; ++i)
	{
		if (i < data.rows / 2)
		{
			responses(i, 0) = 1;//为类别0的置信度
			responses(i, 1) = 0;//为类别1的置信度
		}
		else
		{
			responses(i, 0) = 0;//为类别0的置信度
			responses(i, 1) = 1;//为类别1的置信度
		}
	}
	/*
	//example code for just a single response (regression)
	Mat_<float> responses(data.rows, 1);
	for (int i=0; i<responses.rows; ++i)
	responses(i, 0) = i < responses.rows / 2 ? 0 : 1;
	*/
	//create the neural network
	//创建一个三层的感知器，输入层跟数据维度有关系
	Mat_<int> layerSizes(1, 3);
	layerSizes(0, 0) = data.cols; //输入层中每个样例的维数
	layerSizes(0, 1) = 20; //每个隐藏层中的神经元数目
	layerSizes(0, 2) = responses.cols; //输出层中类别维数（即类别总数）,一般是类别表
	Ptr<ANN_MLP> network = ANN_MLP::create();
	network->setLayerSizes(layerSizes);
	network->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.1, 0.1);
	network->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
	Ptr<TrainData> trainData = TrainData::create(data, ROW_SAMPLE, responses);
	network->train(trainData);
	if (network->isTrained())
	{
		printf("Predict one-vector:\n");
		Mat result;
		network->predict(Mat::ones(1, data.cols, data.type()), result);
		cout << result << endl;
		printf("Predict training data:\n");
		for (int i = 0; i<data.rows; ++i)
		{
			network->predict(data.row(i), result);
			cout << result << endl;
		}
	}
	return 0;
}

int CBusin_OpenCV_ML_Tool::test_ANN_2()
{
	float labels[10][2] = { { 0.9,0.1 },{ 0.1,0.9 },{ 0.9,0.1 },{ 0.1,0.9 },{ 0.9,0.1 },{ 0.9,0.1 },{ 0.1,0.9 },{ 0.1,0.9 },{ 0.9,0.1 },{ 0.9,0.1 } };
	//这里对于样本标记为0.1和0.9而非0和1，主要是考虑到sigmoid函数的输出为一般为0和1之间的数，只有在输入趋近于-∞和+∞才逐渐趋近于0和1，而不可能达到。
	Mat labelsMat(10, 2, CV_32FC1, labels);

	float trainingData[10][2] = { { 11,12 },{ 111,112 },{ 21,22 },{ 211,212 },{ 51,32 },{ 71,42 },{ 441,412 },{ 311,312 },{ 41,62 },{ 81,52 } };
	Mat trainingDataMat(10, 2, CV_32FC1, trainingData);
	Mat layerSizes = (Mat_<int>(1, 5) << 2, 2, 2, 2, 2); //5层：输入层，3层隐藏层和输出层，每层均为两个perceptron



	Ptr<ANN_MLP> ann = ANN_MLP::create();
	ann->setLayerSizes(layerSizes);//
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	//	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, FLT_EPSILON));
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.9);
	Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	ann->train(tData);

	// Data for visual representation  
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);

			Mat responseMat;
			ann->predict(sampleMat, responseMat);
			float* p = responseMat.ptr<float>(0);
			if (p[0] > p[1])
			{
				image.at<Vec3b>(j, i) = green;
			}
			else
			{
				image.at<Vec3b>(j, i) = blue;
			}
		}
	}
	// Show the training data  
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(111, 112), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(211, 212), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(441, 412), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(311, 312), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(11, 12), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(21, 22), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(51, 32), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(71, 42), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(41, 62), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(81, 52), 5, Scalar(255, 255, 255), thickness, lineType);

	imwrite("result.png", image);        // save the image   

	cv::imshow("BP Simple Example", image); // show it to the user  
	cv::waitKey(0);
    cv::destroyAllWindows();
	return 0;
}

int CBusin_OpenCV_ML_Tool::get_labels_and_class_data(const std::string& str_src_imgs_dir,  const std::string& str_class_prefix
		, cv::Mat& mat_trains_dataes, cv::Mat& mat_trains_lables)
{
	std::vector<std::string > vec_src_imgs_pathes;
	int ret = sp_boost::get_files_path_list(str_src_imgs_dir, vec_src_imgs_pathes, ".bmp");
	if (ret)
	{
		std::cout << __FUNCTION__ << " | fail to get files from dir:" << str_src_imgs_dir << endl;
		return ret;
	}
	std::vector<int> vec_labels; 
	cv::Mat mat_element(1, 3, CV_32F); 
	for (const auto& str_src_img_path : vec_src_imgs_pathes)
	{
		//获取标签
		int nLabel = -1;
		ret = get_class_num_from_path(str_src_img_path, str_class_prefix, nLabel);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | fail to get label from path:" << str_src_img_path << endl; 
			return ret;
		}
		//加载图片
		cv::Mat mat_src_bgr = cv::imread(str_src_img_path);
		if (mat_src_bgr.empty())
		{
			std::cout << __FUNCTION__ << " | fail to read image:" << str_src_img_path << endl;
			return -1;
		}
		cv::Mat mat_dst_gray(mat_src_bgr.size(), CV_8UC1, cv::Scalar(0));
		cv::Mat mat_hsv;
		cv::cvtColor(mat_src_bgr, mat_hsv, cv::COLOR_BGR2HSV);
		//将图片中的每个非0像素点对应的hsv放入训练矩阵和标签列表中
		for (int nRow = 0; nRow < mat_src_bgr.rows; ++nRow)
		{
			const cv::Vec3b* pSrc_row_data = mat_src_bgr.ptr<cv::Vec3b>(nRow);
			uchar* pDst_row_data = mat_dst_gray.ptr<uchar>(nRow);
			const cv::Vec3b* pHsv_row_data = mat_hsv.ptr<cv::Vec3b>(nRow);
			for (int nCol = 0; nCol < mat_src_bgr.cols; ++nCol)
			{
				if (false == is_background(pSrc_row_data[nCol]))
				{//不为背景点
					pDst_row_data[nCol] = 255;
					vec_labels.push_back(nLabel);
					mat_element.at<float>(0) = (float)pHsv_row_data[nCol][0];
					mat_element.at<float>(1) = (float)pHsv_row_data[nCol][1];
					mat_element.at<float>(2) = (float)pHsv_row_data[nCol][2];
					mat_trains_dataes.push_back(mat_element);
				}
			}
		}//遍历一个图片文件结束
	}//加载全部图片结束
	//将标签列表转换为矩阵
	mat_trains_lables = cv::Mat(vec_labels).clone();
	return 0;
}

int CBusin_OpenCV_ML_Tool::get_class_num_from_path(const std::string& str_file_path, const std::string& str_class_prefix , int& nLabel)
{
	//查找类别标签
	std::string::size_type nPos_class_prefix = str_file_path.find(str_class_prefix);
	if (nPos_class_prefix == std::string::npos)
	{
		std::cout << __FUNCTION__ << " | path:" << str_file_path << "不含有类别前缀:" << str_class_prefix << endl;
		return -1;
	}
	//查找最后一个'/'
	std::string::size_type nPos_last_slash = str_file_path.find_last_of('/');
	if (nPos_last_slash == std::string::npos)
	{
		nPos_last_slash = str_file_path.find_last_of('\\');
		if (nPos_last_slash == std::string::npos)
		{
			std::cout << __FUNCTION__ << " | path:" << str_file_path << "中不含有\\或/" << endl;
			return -1;
		}
	}
	//标签在文件名中的起始位置
	std::string::size_type nStart_pos_label = nPos_class_prefix + str_class_prefix.length();
	std::string str_lable = str_file_path.substr(nStart_pos_label, nPos_last_slash - nStart_pos_label);
	try
	{
		nLabel = boost::lexical_cast<int>(str_lable);
		return 0;
	}
	catch (std::exception& e)
	{
		std::cout << __FUNCTION__ << " |  Has exception, str_lable:" << str_lable << ", file path:" << str_file_path << endl;
		return -2;
	}

}

int CBusin_OpenCV_ML_Tool::create_gray_with_model(const cv::Ptr<cv::ml::StatModel>& model)
{
	std::string str_src_imgs_dir;
	std::cout << "请输入源彩图目录以构造灰度图:";
	std::cin >> str_src_imgs_dir;

	std::string str_dst_gray_dir;
	std::cout << "请输入存放结果的目录地址:";
	std::cin >> str_dst_gray_dir;
	vector<string> vec_src_img_pathes;
	int ret = sp_boost::get_files_path_list(str_src_imgs_dir, vec_src_img_pathes, ".bmp");
	if (ret)
	{
		std::cout << __FUNCTION__ << " | fail to get bmp files from dir:" << str_src_imgs_dir << std::endl;
		return ret;
	}
	int64  nTotal_tick_count = 0; //总时钟数
	int nTotal_pixel_num = 0; //总的像素数目
	for (const auto& str_src_img_path : vec_src_img_pathes)
	{
		//加载图片
		cv::Mat mat_src_bgr = cv::imread(str_src_img_path);
		if (mat_src_bgr.empty())
		{
			std::cout << __FUNCTION__ << " | fail to read image:" << str_src_img_path << endl;
			return -1;
		}
		//转换为HSV
		cv::Mat mat_hsv;
		cv::cvtColor(mat_src_bgr, mat_hsv, cv::COLOR_BGR2HSV);
		cv::Mat mat_dst_gray(mat_src_bgr.size(), CV_8UC1, cv::Scalar(0));
		const int nImg_width = mat_src_bgr.cols;
		const int nImg_height = mat_src_bgr.rows;

		for (int nRow = 0; nRow < nImg_height; ++nRow)
		{
			const cv::Vec3b* pSrc_bgr_row = mat_src_bgr.ptr<cv::Vec3b>(nRow);
			const cv::Vec3b* pHsv_row_data = mat_hsv.ptr<cv::Vec3b>(nRow);
			uchar* pDst_gray_row = mat_dst_gray.ptr<uchar>(nRow);

			for (int nCol = 0; nCol < nImg_width; ++nCol)
			{
				++nTotal_pixel_num;//总的像素加1
				int64 nStart_tick_count = cv::getTickCount();
				if (!is_background(pSrc_bgr_row[nCol]))
				{
					cv::Mat mat_sample = (cv::Mat_<float>(1, 3) << pHsv_row_data[nCol][0]
					, pHsv_row_data[nCol][1],pHsv_row_data[nCol][2]);
					float r = model->predict(mat_sample);
					if (std::abs(r - 1) < FLT_EPSILON)
					{//类别1
						pDst_gray_row[nCol] = 150;
					}
					else if (std::abs(r - 2) < FLT_EPSILON)
					{//类别2
						pDst_gray_row[nCol] = 150;
					}
					else if (std::abs(r - 3) < FLT_EPSILON)
					{
						pDst_gray_row[nCol] = 255;
					}
					else
					{
						pDst_gray_row[nCol] = 255;
					}

				}
				nTotal_tick_count += (cv::getTickCount() - nStart_tick_count);
			}
		}
		//目标文件路径
		string str_dst_img_path = str_dst_gray_dir + "/" + boost::filesystem::path(str_src_img_path).filename().string();
		if (!cv::imwrite(str_dst_img_path, mat_dst_gray))
		{
			std::cout << __FUNCTION__ << " |fail to write image to file:" << str_dst_img_path << endl;
			return -1;
		}
	}
	//输出平均单帧耗时
	const int nFrame_size = 32 * 2048;
	double nTotal_frame_num = (nTotal_pixel_num * 1.0f / nFrame_size); //总帧数
	double dAvg_cost_for_single_frame =  nTotal_tick_count * 1.0f / (cv::getTickFrequency() * nTotal_frame_num) * 1000; 
	cout << __FUNCTION__ << " | 当帧大小设置为:" << nFrame_size << "时, 单帧平均耗时:" << dAvg_cost_for_single_frame << "ms" << endl;

	return 0;
}
