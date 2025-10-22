package com.mtcnn;

public class Face {
    private float x1;
    private float y1;
    private float x2;
    private float y2;
    private float score;
    private float[] landmarks;
    
    public Face(float x1, float y1, float x2, float y2, float score, float[] landmarks) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.score = score;
        this.landmarks = landmarks;
    }
    
    // Getters
    public float getX1() { return x1; }
    public float getY1() { return y1; }
    public float getX2() { return x2; }
    public float getY2() { return y2; }
    public float getWidth() { return x2 - x1; }
    public float getHeight() { return y2 - y1; }
    public float getScore() { return score; }
    public float[] getLandmarks() { return landmarks; }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Face [");
        sb.append("bbox=(" + x1 + ", " + y1 + ", " + x2 + ", " + y2 + "), ");
        sb.append("score=" + score + ", ");
        sb.append("landmarks=[");
        for (int i = 0; i < landmarks.length; i += 2) {
            if (i > 0) sb.append(", ");
            sb.append("(" + landmarks[i] + ", " + landmarks[i+1] + ")");
        }
        sb.append("]]");
        return sb.toString();
    }
}
