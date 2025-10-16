"""
Configuration class for Face Recognition System
Quản lý tất cả thresholds và parameters cho hệ thống nhận diện khuôn mặt
"""
import cv2

# Get font constant after cv2 is fully imported
TEXT_FONT_DEFAULT = None
try:
    TEXT_FONT_DEFAULT = cv2.FONT_HERSHEY_SIMPLEX
except AttributeError:
    # Fallback if cv2 doesn't have FONT_HERSHEY_SIMPLEX
    TEXT_FONT_DEFAULT = 0  # cv2.FONT_HERSHEY_SIMPLEX typically equals 0

class FaceRecognitionConfig:
    """Configuration class for face recognition system"""
    
    # ==================== QUALITY THRESHOLDS ====================
    # Threshold cho việc đánh giá chất lượng face
    REGISTRATION_QUALITY_THRESHOLD = 0.85  # Cao hơn cho đăng ký (chất lượng tốt)
    INFERENCE_QUALITY_THRESHOLD = 0.70     # Thấp hơn cho nhận diện (linh hoạt hơn)
    
    # ==================== RECOGNITION THRESHOLDS ====================
    # Threshold cho việc nhận diện face
    RECOGNITION_SIMILARITY_THRESHOLD = 0.30  # Cosine similarity threshold
    DUPLICATE_DETECTION_THRESHOLD = 0.60     # Phát hiện duplicate khi đăng ký
    
    # ==================== FACE QUALITY CHECKS ====================
    # Blur detection thresholds
    REGISTRATION_BLUR_THRESHOLD = 100       # Cao hơn cho đăng ký
    INFERENCE_BLUR_THRESHOLD = 80           # Thấp hơn cho nhận diện
    
    # Face size thresholds
    REGISTRATION_MIN_FACE_SIZE = 80         # Kích thước tối thiểu cho đăng ký
    INFERENCE_MIN_FACE_SIZE = 50            # Kích thước tối thiểu cho nhận diện
    
    # Lighting condition thresholds
    REGISTRATION_DARK_RATIO_THRESHOLD = 0.3   # Tỷ lệ pixel tối cho đăng ký
    REGISTRATION_BRIGHT_RATIO_THRESHOLD = 0.3 # Tỷ lệ pixel sáng cho đăng ký
    INFERENCE_DARK_RATIO_THRESHOLD = 0.4      # Linh hoạt hơn cho nhận diện
    INFERENCE_BRIGHT_RATIO_THRESHOLD = 0.4    # Linh hoạt hơn cho nhận diện
    
    # Pose estimation thresholds
    REGISTRATION_POSE_THRESHOLD = 2000      # MSE threshold cho đăng ký
    INFERENCE_POSE_THRESHOLD = 3000         # MSE threshold cho nhận diện
    
    # ==================== MTCNN SETTINGS ====================
    MTCNN_MARGIN = 20
    MTCNN_KEEP_ALL = False
    MTCNN_SELECT_LARGEST = True
    MTCNN_POST_PROCESS = True
    MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]     # P-Net, R-Net, O-Net thresholds
    
    # ==================== TRACKING SETTINGS ====================
    DETECTION_INTERVAL = 5                  # Số frames giữa các lần detect mới
    TRACKER_TYPE = 'CSRT'                   # Loại tracker sử dụng
    
    # ==================== CAMERA SETTINGS ====================
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FACE_SIZE = 160                         # Kích thước face tensor
    
    # ==================== REGISTRATION SETTINGS ====================
    REGISTRATION_SAMPLES = 70               # Số lượng samples cần capture
    REGISTRATION_SKIP_FRAMES = 5            # Skip frames giữa các lần capture
    
    # ==================== DISPLAY SETTINGS ====================
    BBOX_THICKNESS = 6
    TEXT_FONT = TEXT_FONT_DEFAULT
    TEXT_SCALE = 0.8
    TEXT_THICKNESS = 2
    
    # Colors (BGR format)
    COLOR_RECOGNIZED = (0, 255, 0)         # Xanh lá - nhận diện thành công
    COLOR_UNKNOWN = (0, 0, 255)            # Đỏ - không nhận diện được
    COLOR_POOR_QUALITY = (0, 255, 255)     # Vàng - chất lượng kém
    COLOR_TRACKING = (0, 0, 255)           # Đỏ - đang track
    COLOR_WHITE = (255, 255, 255)          # Trắng - text thông tin
    
    @classmethod
    def get_quality_thresholds(cls, mode='inference'):
        """Get quality thresholds based on mode"""
        if mode == 'registration':
            return {
                'blur_threshold': cls.REGISTRATION_BLUR_THRESHOLD,
                'min_face_size': cls.REGISTRATION_MIN_FACE_SIZE,
                'dark_ratio_threshold': cls.REGISTRATION_DARK_RATIO_THRESHOLD,
                'bright_ratio_threshold': cls.REGISTRATION_BRIGHT_RATIO_THRESHOLD,
                'pose_threshold': cls.REGISTRATION_POSE_THRESHOLD,
                'quality_threshold': cls.REGISTRATION_QUALITY_THRESHOLD
            }
        else:  # inference
            return {
                'blur_threshold': cls.INFERENCE_BLUR_THRESHOLD,
                'min_face_size': cls.INFERENCE_MIN_FACE_SIZE,
                'dark_ratio_threshold': cls.INFERENCE_DARK_RATIO_THRESHOLD,
                'bright_ratio_threshold': cls.INFERENCE_BRIGHT_RATIO_THRESHOLD,
                'pose_threshold': cls.INFERENCE_POSE_THRESHOLD,
                'quality_threshold': cls.INFERENCE_QUALITY_THRESHOLD
            }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 FACE RECOGNITION CONFIGURATION:")
        print("=" * 50)
        print(f"Registration Quality Threshold: {cls.REGISTRATION_QUALITY_THRESHOLD}")
        print(f"Inference Quality Threshold: {cls.INFERENCE_QUALITY_THRESHOLD}")
        print(f"Recognition Similarity Threshold: {cls.RECOGNITION_SIMILARITY_THRESHOLD}")
        print(f"Duplicate Detection Threshold: {cls.DUPLICATE_DETECTION_THRESHOLD}")
        print("=" * 50)
