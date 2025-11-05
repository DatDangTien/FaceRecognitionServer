package com.mtcnn;

public class TestMTCNN {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java -cp <classpath> com.mtcnn.TestMTCNN <model_path> <image_path>");
            System.exit(1);
        }
        
        String modelPath = args[0];
        String imagePath = args[1];
        
        MTCNNDetector detector = new MTCNNDetector();
        Face[] faces = detector.detectFaces(modelPath, imagePath);
        
        System.out.println("Detected " + faces.length + " faces:");
        for (int i = 0; i < faces.length; i++) {
            System.out.println("Face " + (i+1) + ": " + faces[i]);
        }
    }
}
