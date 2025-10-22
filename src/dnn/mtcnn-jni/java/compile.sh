#!/bin/bash

# Create directories
mkdir -p classes
mkdir -p ../cpp

# Compile Java classes
javac -d classes com/mtcnn/*.java

# Generate JNI header
javac -h ../cpp -d classes com/mtcnn/*.java

# Create JAR file
jar cf mtcnn.jar -C classes .

echo "Java compilation and JNI header generation completed."
