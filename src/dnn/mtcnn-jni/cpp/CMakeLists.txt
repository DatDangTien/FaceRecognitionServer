cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0167 NEW)

project(mtcnn_jni)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tìm các thư viện cần thiết
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(JNI REQUIRED)

# Thêm thư mục include
include_directories(
    ${JNI_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

# Thêm các tệp nguồn
set(SOURCES
    mtcnn_jni.cpp
    ../../mtcnn/detector.cpp
    ../../mtcnn/onet.cpp
    ../../mtcnn/pnet.cpp
    ../../mtcnn/rnet.cpp
)

# Tạo thư viện động
add_library(mtcnn_jni SHARED ${SOURCES})

# Liên kết với các thư viện cần thiết
target_link_libraries(mtcnn_jni ${OpenCV_LIBS} ${Boost_LIBRARIES} ${JNI_LIBRARIES})

# Đặt tên đầu ra cho macOS
set_target_properties(mtcnn_jni PROPERTIES
    PREFIX "lib"
    SUFFIX ".dylib"
)
