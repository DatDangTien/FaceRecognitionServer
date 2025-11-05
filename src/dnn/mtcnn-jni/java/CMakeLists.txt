cmake_minimum_required(VERSION 3.10)

project(mtcnn_java)

# Tạo thư mục cho mã nguồn Java
set(JAVA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/com/mtcnn)
set(JAVA_CLASS_DIR ${CMAKE_CURRENT_BINARY_DIR}/classes)

# Tạo thư mục đầu ra
file(MAKE_DIRECTORY ${JAVA_CLASS_DIR})

# Tìm tất cả các tệp Java
file(GLOB JAVA_SOURCES ${JAVA_SOURCE_DIR}/*.java)

# Biên dịch mã Java và tạo header JNI
add_custom_command(
    OUTPUT ${JAVA_CLASS_DIR}/com/mtcnn/TestMTCNN.class ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/com_mtcnn_MTCNNDetector.h
    COMMAND javac -d ${JAVA_CLASS_DIR} -h ${CMAKE_CURRENT_SOURCE_DIR}/../cpp ${JAVA_SOURCES}
    DEPENDS ${JAVA_SOURCES}
    COMMENT "Compiling Java sources and generating JNI header"
)

# Tạo tệp JAR
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mtcnn.jar
    COMMAND jar cf mtcnn.jar -C ${JAVA_CLASS_DIR} .
    DEPENDS ${JAVA_CLASS_DIR}/com/mtcnn/TestMTCNN.class
    COMMENT "Creating JAR file"
)

# Tạo mục tiêu
add_custom_target(mtcnn_java ALL DEPENDS 
    ${CMAKE_CURRENT_BINARY_DIR}/mtcnn.jar
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/com_mtcnn_MTCNNDetector.h
)