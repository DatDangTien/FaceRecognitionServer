cmake_minimum_required(VERSION 3.10)
project(Postgres)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PostgreSQL REQUIRED)
include_directories(${PostgreSQL_INCLUDE_DIRS})

find_library(PQXX_LIB pqxx REQUIRED)

add_executable(test_connect test_connection.cpp postgres.cpp)
target_link_libraries(test_connect ${PostgreSQL_LIBRARIES} ${PQXX_LIB})

add_executable(migrate_data migrate_data.cpp postgres.cpp)
target_link_libraries(migrate_data ${PostgreSQL_LIBRARIES} ${PQXX_LIB})
