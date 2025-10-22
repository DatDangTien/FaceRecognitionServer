#include <string>
#include <vector>
#include <sstream>

std::string vec2pgvector(const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ",";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<float> pgvector2vec(const std::string& vec) {
    std::string clean_vec = vec.substr(1, vec.size() - 2);
    std::istringstream iss(clean_vec);
    std::vector<float> result;
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stof(token));
    }
    return result;
}