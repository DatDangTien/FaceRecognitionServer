#!/bin/bash

# Thiết lập biến môi trường Java
export JAVA_HOME=$(brew --prefix)/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
export JDK_HOME=$JAVA_HOME

echo "=== Sử dụng Java từ: $JAVA_HOME ==="
java -version
javac -version

# Thư mục gốc của dự án
PROJECT_DIR=$(pwd)

# Biên dịch Java
echo "=== Biên dịch Java ==="
cd $PROJECT_DIR/java
mkdir -p classes
javac -d classes com/mtcnn/*.java

# Tạo header JNI
echo "=== Tạo header JNI ==="
javac -h ../cpp -d classes com/mtcnn/*.java

# Biên dịch C++
echo "=== Biên dịch C++ ==="
cd $PROJECT_DIR/cpp
mkdir -p build
cd build

# Chạy CMake với đường dẫn JNI
cmake -DJNI_INCLUDE_DIRS="$JDK_HOME/include:$JDK_HOME/include/darwin" ..
make

echo "=== Quá trình biên dịch hoàn tất ==="
echo "Để chạy ứng dụng, sử dụng lệnh:"
echo "java -Djava.library.path=$PROJECT_DIR/cpp/build -cp $PROJECT_DIR/java/classes com.mtcnn.TestMTCNN <đường_dẫn_đến_models> <đường_dẫn_đến_ảnh>"
