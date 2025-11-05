package com.mtcnn;

public class MTCNNDetector {
    static {
        System.loadLibrary("mtcnn_jni");
    }
    
    // Phương thức native để phát hiện khuôn mặt
    public native Face[] detectFaces(String modelPath, String imagePath, float minFaceSize, float scaleFactor);
    
    // Phương thức tiện ích với các giá trị mặc định
    public Face[] detectFaces(String modelPath, String imagePath) {
        return detectFaces(modelPath, imagePath, 20.0f, 0.709f);
    }
}
