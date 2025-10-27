#!/usr/bin/env python3
"""
Test script to verify OpenCV display works without Qt5 threading issues
"""
import cv2
import numpy as np
import os

# Apply the same fixes as in face_capture_noSave.py
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
cv2.setUseOptimized(True)

def test_opencv_display():
    print("Testing OpenCV display with Qt5 threading fixes...")
    
    # Create a test image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # White background
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
    cv2.circle(img, (300, 150), 50, (0, 0, 255), -1)
    cv2.putText(img, "OpenCV Test - Fixed", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Created test image successfully")
    
    try:
        # Test window creation and display
        cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Window", img)
        print("Successfully created window and displayed image")
        print("Press any key to exit...")
        
        # Test waitKey with timeout (like in the face recognition code)
        key = cv2.waitKey(1000)  # Wait 1 second
        print(f"waitKey returned: {key}")
        
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f"Error displaying image: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting OpenCV display test with Qt5 fixes...")
    success = test_opencv_display()
    print("Test completed. Success:", success)