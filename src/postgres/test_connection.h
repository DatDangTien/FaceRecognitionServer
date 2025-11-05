#ifndef TEST_CONNECTION_H
#define TEST_CONNECTION_H

#include <pqxx/pqxx>
#include <vector>
#include <string>

// Convert a vector of floats to PostgreSQL vector format string
std::string vec2pgvector(const std::vector<float>& vec);

#endif // TEST_CONNECTION_H

