#!/bin/bash

# Thiết lập biến môi trường Java
export JAVA_HOME=$(brew --prefix)/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
export JDK_HOME=$JAVA_HOME

# Kiểm tra tham số
if [ $# -lt 2 ]; then
    echo "Sử dụng: $0 <đường_dẫn_đến_models> <đường_dẫn_đến_ảnh>"
    exit 1
fi

MODEL_PATH=$1
IMAGE_PATH=$2

# Thư mục gốc của dự án
PROJECT_DIR=$(pwd)

# Thiết lập đường dẫn thư viện
export LD_LIBRARY_PATH=$PROJECT_DIR/cpp/build:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PROJECT_DIR/cpp/build:$DYLD_LIBRARY_PATH

# Chạy ứng dụng Java
echo "=== Chạy ứng dụng MTCNN JNI ==="
java -Djava.library.path=$PROJECT_DIR/cpp/build -cp $PROJECT_DIR/java/classes com.mtcnn.TestMTCNN $MODEL_PATH $IMAGE_PATH
