#pragma once

#include <string>
#include <vector>

/**
 * Get a list of image files from a directory
 * @param data_dir Path to the directory containing image files
 * @return Vector of image filenames found in the directory
 */
std::vector<std::string> get_image_files(const std::string& data_dir);


