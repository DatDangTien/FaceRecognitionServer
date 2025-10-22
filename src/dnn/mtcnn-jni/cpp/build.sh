#!/bin/bash

# Check for JDK_HOME environment variable
if [ -z "$JDK_HOME" ]; then
    echo "JDK_HOME environment variable is not set. Please set it to your JDK installation directory."
    echo "For example: export JDK_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_xxx.jdk/Contents/Home"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Run CMake with JNI include path
cmake -DJNI_INCLUDE_DIRS="$JDK_HOME/include;$JDK_HOME/include/darwin" ..

# Build the library
make

echo "C++ library build completed."
