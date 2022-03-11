#ifndef COMMON_FILE_HELPER_H
#define COMMON_FILE_HELPER_H

#include "common/config.h"
#include "common/common.h"

template<typename DataType>
void Save(std::ofstream& f, const DataType& val);
template<typename DataType>
const DataType Load(std::ifstream& f);
template<typename DataType>
void SaveVector(std::ofstream& f, const std::vector<DataType>& vals) {
    const int num = static_cast<int>(vals.size());
    Save<int>(f, num);
    for (int i = 0; i < num; ++i) Save<DataType>(f, vals[i]);
}

template<typename DataType>
void LoadVector(std::ifstream& f, std::vector<DataType>& vals) {
    const int num = Load<int>(f);
    CheckError(num >= 0, "Load<int>: invalid num: " + std::to_string(num));
    vals.resize(num);
    for (int i = 0; i < num; ++i) vals[i] = Load<DataType>(f);
}

// Replace all '\' with '/'.
const std::string RegularizeFilePath(const std::string& path);
// Concatenate folder and file paths.
const std::string AppendFileToPath(const std::string& folder,
                                   const std::string& file_name);
const std::string AppendFolderToPath(const std::string& folder,
                                     const std::string& subfolder);
const std::string GetParentFolder(const std::string path);
void PrepareToCreateFile(const std::string& file_path);
const bool FileExist(const std::string& file_path);

#endif