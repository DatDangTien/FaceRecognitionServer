import cv2
import numpy as np

def test_opencv():
    print("OpenCV version:", cv2.__version__)
    
    # Create a simple test image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # White background
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
    cv2.circle(img, (300, 150), 50, (0, 0, 255), -1)
    cv2.putText(img, "OpenCV Test", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Created test image successfully")
    
    try:
        # Try to create a window and display the image
        cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Window", img)
        print("Successfully created window and displayed image")
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print("Error displaying image:", str(e))
        return False

if __name__ == "__main__":
    print("Starting OpenCV test...")
    success = test_opencv()
    print("Test completed. Success:", success)
