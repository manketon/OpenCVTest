/******************************************************************
* Copyright (c) 2016-2017,HIIRI Inc.
* All rights reserved.
* @file: File_System_Tool.h
* @brief: 使用boost::file_system提供文件系统相关的操作。不提供单列对象。只提供一下方法。
* @author:	minglu2
* @version: 1.0
* @date: 2018/03/23
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本	<th>日期		<th>作者	<th>备注 </tr>
*  <tr> <td>1.0	    <td>2018/03/23	<td>minglu	<td>Create head file </tr>
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
#include <boost/filesystem.hpp>
#include <string>
namespace fs = boost::filesystem;
using std::string;
//基于boost库而写的工具
namespace sp_boost
{
	//宏定义
//************************************
// Method:    get_files_list
// Brief:  获取指定目录中以指定串结尾的文件的路径列表
// Access:    public 
// Returns:   int 0:成功 非0: 错误码
// Qualifier:
// Parameter: const string & dir_str -[in]  
// Parameter: STL_Container_Type & filename_container -[out]  
// Parameter: const string& suffix_str -[in] 文件名后缀，如".jpg"或者".xml"等 ，为空表示获取所有文件
// Parameter: bool recursive -[in] 是否搜索子目录 
// Parameter: int max_file -[in]  最大文件数目
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
	//判定容器中的文件数目是否已经达到
	if (filename_container.size() >= max_file )
	{
		return 0;
	}
	fs::directory_iterator end_iter;
	for (fs::directory_iterator iter(path); iter != end_iter; ++iter)
	{
		//查看当前目录中的文件
		if (fs::is_regular_file(iter->status()) && filename_container.size() < max_file)
		{//是文件且数量不够且以设定的后缀名一样
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
			//查看当前目录的子目录
			if (fs::is_directory(iter->status()) && filename_container.size() < max_file)
			{//是目录且数量不够
				ret = get_files_path_list(iter->path().string(), filename_container, suffix_str, recursive, max_file);
				if (ret != 0)
				{//出错则退出
					return ret;
				}
			}
		}
	}
	return ret;
}

} //end namespace sp_boost

//函数原型定义
