#!/bin/bash

# Check for JDK_HOME environment variable
if [ -z "$JDK_HOME" ]; then
    echo "JDK_HOME environment variable is not set. Please set it to your JDK installation directory."
    echo "For example: export JDK_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_xxx.jdk/Contents/Home"
    exit 1
fi

# Check if model path and image path are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <path_to_models_dir> <path_to_image>"
    exit 1
fi

MODEL_PATH=$1
IMAGE_PATH=$2

# Set library path
export LD_LIBRARY_PATH=./cpp/build:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=./cpp/build:$DYLD_LIBRARY_PATH

# Run the Java application
java -Djava.library.path=./cpp/build -cp ./java/build/mtcnn.jar com.mtcnn.TestMTCNN $MODEL_PATH $IMAGE_PATH

echo "Test completed."
