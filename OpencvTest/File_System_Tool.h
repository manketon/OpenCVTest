/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: File_System_Tool.h
* @brief: ʹ��boost::file_system�ṩ�ļ�ϵͳ��صĲ��������ṩ���ж���ֻ�ṩһ�·�����
* @author:	minglu2
* @version: 1.0
* @date: 2018/03/23
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾	<th>����		<th>����	<th>��ע </tr>
*  <tr> <td>1.0	    <td>2018/03/23	<td>minglu	<td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once

#ifdef __cplusplus  
extern "C" {  
//����C���Խӿڡ������ͷ�ļ�
#endif  
#ifdef __cplusplus  
}  
#endif  
//����C++ͷ�ļ������Ǳ�׼��ͷ�ļ���������Ŀͷ�ļ�
#include <boost/filesystem.hpp>
#include <string>
namespace fs = boost::filesystem;
using std::string;
//����boost���д�Ĺ���
namespace sp_boost
{
	//�궨��
//************************************
// Method:    get_files_list
// Brief:  ��ȡָ��Ŀ¼����ָ������β���ļ���·���б�
// Access:    public 
// Returns:   int 0:�ɹ� ��0: ������
// Qualifier:
// Parameter: const string & dir_str -[in]  
// Parameter: STL_Container_Type & filename_container -[out]  
// Parameter: const string& suffix_str -[in] �ļ�����׺����".jpg"����".xml"�� ��Ϊ�ձ�ʾ��ȡ�����ļ�
// Parameter: bool recursive -[in] �Ƿ�������Ŀ¼ 
// Parameter: int max_file -[in]  ����ļ���Ŀ
//************************************
template<class STL_Container_Type>
int get_files_path_list(const string& dir_str, STL_Container_Type& filename_container, const string& suffix_str = "", bool recursive = true, size_t max_file = INT_MAX);

template<class STL_Container_Type>
int get_files_path_list(const string& dir_str, STL_Container_Type& filename_container, const string& suffix_str /*= ""*/, bool recursive /*= true*/, size_t max_file /*= INT_MAX*/)
{
	int ret = 0;
	fs::path path(dir_str);
	if (!fs::exists(path))
	{
		printf("func:%s | Do not find directory:%s", __FUNCTION__, dir_str.c_str());
		ret = 10115;
		return ret;
	}
	//�ж������е��ļ���Ŀ�Ƿ��Ѿ��ﵽ
	if (filename_container.size() >= max_file )
	{
		return 0;
	}
	fs::directory_iterator end_iter;
	for (fs::directory_iterator iter(path); iter != end_iter; ++iter)
	{
		//�鿴��ǰĿ¼�е��ļ�
		if (fs::is_regular_file(iter->status()) && filename_container.size() < max_file)
		{//���ļ����������������趨�ĺ�׺��һ��
			if (suffix_str.empty())
			{
				filename_container.push_back(iter->path().string());
			}
			else
			{
				if ((fs::extension(*iter) == suffix_str))
				{
					filename_container.push_back(iter->path().string());
				}
			}	
		}

		if (recursive)
		{
			//�鿴��ǰĿ¼����Ŀ¼
			if (fs::is_directory(iter->status()) && filename_container.size() < max_file)
			{//��Ŀ¼����������
				ret = get_files_path_list(iter->path().string(), filename_container, suffix_str, recursive, max_file);
				if (ret != 0)
				{//�������˳�
					return ret;
				}
			}
		}
	}
	return ret;
}

} //end namespace sp_boost

//����ԭ�Ͷ���
