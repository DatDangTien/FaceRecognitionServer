import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
import time
import logging
from face_config import FaceRecognitionConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

frame_size = (640, 480)
DATA_PATH = './data'

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
logger.info(f"Khởi tạo thiết bị: {device}")

# Tracker state class để lưu thông tin nhận diện
class TrackerState:
    def __init__(self, tracker_id):
        self.tracker_id = tracker_id
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.recognition_result = None  # {"name": str, "confidence": float, "status": str}
        self.last_inference_frame = 0
        self.stable_count = 0  # Số frame liên tiếp có kết quả ổn định
        self.is_stable = False  # Đã ổn định chưa
        self.consecutive_failures = 0  # Số lần inference thất bại liên tiếp
        
    def should_run_inference(self, current_frame, inference_interval=5):
        """Kiểm tra có nên chạy inference không"""
        # Nếu chưa có kết quả hoặc không ổn định, thì inference thường xuyên hơn
        if self.recognition_result is None or not self.is_stable:
            return (current_frame - self.last_inference_frame) >= max(1, inference_interval // 2)
        
        # Nếu đã ổn định, inference ít hơn
        return (current_frame - self.last_inference_frame) >= inference_interval
    
    def update_recognition_result(self, name, confidence, status):
        """Cập nhật kết quả nhận diện và kiểm tra tính ổn định"""
        new_result = {"name": name, "confidence": confidence, "status": status}
        
        # Kiểm tra tính ổn định
        if (self.recognition_result is not None and 
            self.recognition_result["name"] == name and 
            abs(self.recognition_result["confidence"] - confidence) < 10.0):  # Chênh lệch < 10%
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.is_stable = False
        
        # Coi là ổn định sau 3 lần kết quả giống nhau
        if self.stable_count >= 3:
            self.is_stable = True
            
        self.recognition_result = new_result
        self.consecutive_failures = 0
        
    def mark_inference_failure(self):
        """Đánh dấu inference thất bại"""
        self.consecutive_failures += 1
        # Nếu thất bại quá nhiều lần, reset trạng thái
        if self.consecutive_failures >= 5:
            self.recognition_result = None
            self.is_stable = False
            self.stable_count = 0

def load_faceslist():
    """Load existing embeddings and usernames"""
    embeddings = None
    usernames = None
    
    # Load embeddings
    if device.type == 'cpu':
        embedding_file = os.path.join(DATA_PATH, "faceslistCPU.pth")
    else:
        embedding_file = os.path.join(DATA_PATH, "faceslist.pth")
    
    if os.path.exists(embedding_file):
        embeddings = torch.load(embedding_file)
        logger.info(f"✅ Đã load embeddings: {embeddings.shape}")
        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        logger.error(f"❌ Không tìm thấy file embeddings: {embedding_file}")
    
    # Load usernames
    username_file = os.path.join(DATA_PATH, "usernames.npy")
    if os.path.exists(username_file):
        usernames = np.load(username_file)
        logger.info(f"✅ Đã load usernames: {usernames}")
        print(f"Loaded usernames: {usernames}")
    else:
        logger.error(f"❌ Không tìm thấy file usernames: {username_file}")
    
    return embeddings, usernames

def validate_face_quality(face_tensor, frame_roi, mode='inference'):
    """Kiểm tra chất lượng face trước khi nhận diện"""
    # Get thresholds from config based on mode
    thresholds = FaceRecognitionConfig.get_quality_thresholds(mode)
    
    quality_score = 0.0
    checks_passed = 0
    total_checks = 4
    quality_details = {}
    
    # Convert tensor to numpy for OpenCV operations
    if isinstance(face_tensor, torch.Tensor):
        # face_tensor shape: [3, 160, 160] -> [160, 160, 3]
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_np = ((face_np + 1) * 127.5).astype(np.uint8)  # Denormalize [-1,1] to [0,255]
    else:
        face_np = frame_roi
    
    # 1. Blur Detection (Laplacian variance)
    gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY) if len(face_np.shape) == 3 else face_np
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_pass = blur_score > thresholds['blur_threshold']
    quality_details['blur_score'] = blur_score
    quality_details['blur_threshold'] = thresholds['blur_threshold']
    quality_details['blur_pass'] = blur_pass
    
    if blur_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # 2. Face Size Validation
    face_height, face_width = gray.shape[:2]
    size_pass = face_height >= thresholds['min_face_size'] and face_width >= thresholds['min_face_size']
    quality_details['face_size'] = (face_width, face_height)
    quality_details['min_face_size'] = thresholds['min_face_size']
    quality_details['size_pass'] = size_pass
    
    if size_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # 3. Lighting Condition (histogram analysis)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[200:])
    total_pixels = face_height * face_width
    
    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels
    lighting_pass = dark_ratio < thresholds['dark_ratio_threshold'] and bright_ratio < thresholds['bright_ratio_threshold']
    
    quality_details['dark_ratio'] = dark_ratio
    quality_details['bright_ratio'] = bright_ratio
    quality_details['dark_ratio_threshold'] = thresholds['dark_ratio_threshold']
    quality_details['bright_ratio_threshold'] = thresholds['bright_ratio_threshold']
    quality_details['lighting_pass'] = lighting_pass
    
    if lighting_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # 4. Pose Estimation (basic symmetry check)
    mid_point = face_width // 2
    left_half = gray[:, :mid_point]
    right_half = gray[:, mid_point:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_resized = cv2.resize(left_half, (min_width, face_height))
    right_resized = cv2.resize(right_half_flipped, (min_width, face_height))
    
    mse = np.mean((left_resized.astype(float) - right_resized.astype(float)) ** 2)
    pose_pass = mse < thresholds['pose_threshold']
    quality_details['pose_mse'] = mse
    quality_details['pose_threshold'] = thresholds['pose_threshold']
    quality_details['pose_pass'] = pose_pass
    
    if pose_pass:
        quality_score += 0.25
        checks_passed += 1
    
    is_good_quality = quality_score >= thresholds['quality_threshold']
    
    # Log chi tiết về quality check (chỉ khi debug)
    logger.debug(f"🔍 QUALITY CHECK - Score: {quality_score:.3f}/{thresholds['quality_threshold']:.3f}")
    logger.debug(f"   📏 Size: {quality_details['face_size']} (min: {quality_details['min_face_size']}) - {'✅' if size_pass else '❌'}")
    logger.debug(f"   🌫️  Blur: {blur_score:.1f} (min: {thresholds['blur_threshold']}) - {'✅' if blur_pass else '❌'}")
    logger.debug(f"   💡 Lighting: Dark={dark_ratio:.3f}, Bright={bright_ratio:.3f} - {'✅' if lighting_pass else '❌'}")
    logger.debug(f"   👤 Pose MSE: {mse:.1f} (max: {thresholds['pose_threshold']}) - {'✅' if pose_pass else '❌'}")
    logger.debug(f"   ✅ Checks passed: {checks_passed}/{total_checks} - {'PASS' if is_good_quality else 'FAIL'}")
    
    return is_good_quality, quality_score

def inference(model, face_tensor, local_embeds, threshold=None, usernames=None):
    """Improved inference with better threshold and preprocessing"""
    if face_tensor is None:
        logger.warning("⚠️ Face tensor is None, skipping inference")
        return -1, -1, 0.0
    
    # Use config threshold if not provided
    if threshold is None:
        threshold = FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD
    
    # Convert threshold to similarity threshold (since we're using cosine similarity)
    similarity_threshold = 1 - threshold
    
    logger.debug(f"🎯 INFERENCE - Distance threshold: {threshold}, Similarity threshold: {similarity_threshold}")
    
    with torch.no_grad():
        # Use the same preprocessing as in face_capture_noSave.py
        detect_embeds = model(face_tensor.unsqueeze(0).to(device))
    
    # Calculate cosine similarity (more robust than euclidean distance)
    # Normalize embeddings
    detect_embeds_norm = torch.nn.functional.normalize(detect_embeds, p=2, dim=1)
    local_embeds_norm = torch.nn.functional.normalize(local_embeds, p=2, dim=1)
    
    # Calculate cosine similarity
    similarity = torch.mm(detect_embeds_norm, local_embeds_norm.t())
    
    # Get the best match
    max_sim, embed_idx = torch.max(similarity, dim=1)
    
    # Convert similarity to distance (1 - similarity) and percentage
    distance = 1 - max_sim
    confidence_percentage = max_sim.item() * 100  # Similarity as percentage
    
    # Log all similarities for debugging
    if usernames is not None and len(usernames) > 0:
        logger.debug(f"📊 SIMILARITY SCORES:")
        for i, sim in enumerate(similarity[0]):
            sim_percent = sim.item() * 100
            dist = 1 - sim.item()
            logger.debug(f"   {usernames[i]}: similarity={sim_percent:.2f}%, distance={dist:.3f}")
        
        best_match_name = usernames[embed_idx.item()] if embed_idx.item() < len(usernames) else "Unknown"
        logger.debug(f"🏆 BEST MATCH: {best_match_name}")
        logger.debug(f"   📈 Confidence: {confidence_percentage:.2f}%")
        logger.debug(f"   📏 Distance: {distance.item():.3f}")
        logger.debug(f"   🎯 Threshold: distance={threshold} (similarity={similarity_threshold})")
    
    # Decision logic
    if distance > threshold:
        logger.debug(f"❌ REJECTED - Distance {distance.item():.3f} > threshold {threshold}")
        logger.debug(f"   📉 Confidence {confidence_percentage:.1f}% < required {(1-threshold)*100:.1f}%")
        return -1, -1, confidence_percentage
    else:
        logger.debug(f"✅ RECOGNIZED - Distance {distance.item():.3f} <= threshold {threshold}")
        logger.debug(f"   📈 Confidence {confidence_percentage:.1f}% >= required {(1-threshold)*100:.1f}%")
        return embed_idx, distance, confidence_percentage

if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    
    logger.info("🚀 Khởi động Face Recognition System")
    
    # Load model and data
    logger.info("📦 Đang load model...")
    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()
    logger.info("✅ Model loaded successfully")

    # Use same MTCNN settings as face_capture_noSave.py
    logger.info("🔧 Khởi tạo MTCNN...")
    mtcnn = MTCNN(
        margin=FaceRecognitionConfig.MTCNN_MARGIN,
        keep_all=FaceRecognitionConfig.MTCNN_KEEP_ALL,
        select_largest=FaceRecognitionConfig.MTCNN_SELECT_LARGEST,
        post_process=FaceRecognitionConfig.MTCNN_POST_PROCESS,
        device=device
    )
    logger.info("✅ MTCNN initialized")

    # Load embeddings and usernames
    logger.info("📂 Đang load dữ liệu face...")
    embeddings, names = load_faceslist()
    if embeddings is None or names is None:
        logger.error("❌ Không tìm thấy dữ liệu face. Vui lòng đăng ký user trước.")
        print("❌ Không tìm thấy dữ liệu face. Vui lòng đăng ký user trước.")
        exit()
    
    logger.info(f"✅ Đã load {len(names)} users: {', '.join(names)}")

    logger.info("📹 Khởi tạo camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FaceRecognitionConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FaceRecognitionConfig.CAMERA_HEIGHT)
    logger.info("✅ Camera initialized")
    
    # Khởi tạo tracker states
    tracker_states = []  # List of TrackerState objects
    detection_interval = FaceRecognitionConfig.DETECTION_INTERVAL
    recognition_interval = 3  # Chỉ inference 5 frame 1 lần
    frame_count = 0
    
    # Recognition statistics
    total_faces_processed = 0
    total_inferences_run = 0  # Số lần thực sự chạy inference
    successful_recognitions = 0
    quality_rejections = 0
    similarity_rejections = 0
    cached_results = 0  # Số lần dùng kết quả cache
    
    print("🎭 Bắt đầu Face Recognition...")
    print(f"⚡ Tối ưu hóa: Inference mỗi {recognition_interval} frame")
    print("Nhấn ESC để thoát")
    logger.info("🎭 Face Recognition started")
    logger.info(f"⚡ Performance optimization: Recognition interval = {recognition_interval} frames")
    
    # Hiển thị config hiện tại
    FaceRecognitionConfig.print_config()
    logger.info(f"⚙️ Recognition threshold: {FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD}")
    logger.info(f"⚙️ Required confidence: {(1-FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD)*100:.1f}%")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces mới hoặc khi cần làm mới trackers
        if len(tracker_states) == 0 or frame_count % detection_interval == 0:
            boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
            if boxes is not None:
                logger.debug(f"🔍 Detected {len(boxes)} faces in frame {frame_count}")
                
                # Tạo trackers mới và reset states
                new_tracker_states = []
                for i, box in enumerate(boxes):
                    tracker_state = TrackerState(tracker_id=len(new_tracker_states))
                    bbox = (int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1]))
                    success = tracker_state.tracker.init(frame, bbox)
                    if success:
                        new_tracker_states.append(tracker_state)
                        logger.debug(f"   Tracker {i}: bbox={bbox}")
                
                tracker_states = new_tracker_states
        
        # Update trackers và nhận diện với optimization
        active_trackers = []
        for tracker_state in tracker_states:
            success, bbox = tracker_state.tracker.update(frame)
            if success:
                active_trackers.append(tracker_state)
                # Vẽ bbox từ tracker
                x, y, w, h = [int(v) for v in bbox]
                
                # Quyết định có chạy inference không
                should_infer = tracker_state.should_run_inference(frame_count, recognition_interval)
                
                if should_infer:
                    # Lấy face từ bbox để nhận diện
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        
                        # Extract face tensor using MTCNN (same as capture)
                        try:
                            face_tensor = mtcnn(face_rgb)
                        except RuntimeError as e:
                            if "expected a non-empty list of Tensors" in str(e):
                                logger.debug("No face detected by MTCNN, skipping inference for this tracker.")
                                face_tensor = None
                            else:
                                raise e
                        if face_tensor is not None:
                            total_faces_processed += 1
                            total_inferences_run += 1  # Đếm inference thực sự
                            tracker_state.last_inference_frame = frame_count
                            
                            logger.debug(f"🎯 Running inference for tracker {tracker_state.tracker_id} (frame {frame_count})")
                            
                            # Validate face quality
                            is_good_quality, quality_score = validate_face_quality(face_tensor, face_rgb, mode='inference')
                            
                            if is_good_quality:
                                # Perform recognition
                                idx, distance, confidence_percentage = inference(model, face_tensor, embeddings, usernames=names)
                                
                                if idx != -1:
                                    # Update tracker state với kết quả nhận diện
                                    tracker_state.update_recognition_result(
                                        name=names[idx], 
                                        confidence=confidence_percentage, 
                                        status="recognized"
                                    )
                                    successful_recognitions += 1
                                    logger.info(f"✅ RECOGNIZED: {names[idx]} with {confidence_percentage:.1f}% confidence (Tracker {tracker_state.tracker_id})")
                                else:
                                    # Update tracker state với unknown
                                    tracker_state.update_recognition_result(
                                        name="Unknown", 
                                        confidence=confidence_percentage, 
                                        status="unknown"
                                    )
                                    similarity_rejections += 1
                                    logger.info(f"❌ UNKNOWN: Confidence {confidence_percentage:.1f}% < required {(1-FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD)*100:.1f}% (Tracker {tracker_state.tracker_id})")
                            else:
                                # Update tracker state với poor quality
                                tracker_state.update_recognition_result(
                                    name="Poor Quality", 
                                    confidence=quality_score * 100, 
                                    status="poor_quality"
                                )
                                quality_rejections += 1
                                logger.info(f"⚠️ POOR QUALITY: Score {quality_score:.2f} < required threshold (Tracker {tracker_state.tracker_id})")
                        else:
                            tracker_state.mark_inference_failure()
                else:
                    # Sử dụng kết quả cached
                    cached_results += 1
                    logger.debug(f"📋 Using cached result for tracker {tracker_state.tracker_id}")
                
                # Vẽ kết quả trên frame (dùng kết quả cached hoặc mới)
                if tracker_state.recognition_result:
                    result = tracker_state.recognition_result
                    
                    if result["status"] == "recognized":
                        # Green box for recognized face
                        color = FaceRecognitionConfig.COLOR_RECOGNIZED
                        text = f"{result['name']} ({result['confidence']:.1f}%)"
                        if tracker_state.is_stable:
                            text += " ✓"  # Indicate stable recognition
                        
                    elif result["status"] == "unknown":
                        # Red box for unknown face
                        color = FaceRecognitionConfig.COLOR_UNKNOWN
                        text = f"Unknown ({result['confidence']:.1f}%)"
                        
                    else:  # poor_quality
                        # Yellow box for poor quality
                        color = FaceRecognitionConfig.COLOR_POOR_QUALITY
                        text = f"Poor Quality ({result['confidence']:.1f})"
                    
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, FaceRecognitionConfig.BBOX_THICKNESS)
                    frame = cv2.putText(frame, text, (x, y-10), 
                                      FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE, 
                                      color, FaceRecognitionConfig.TEXT_THICKNESS)
                else:
                    # Chưa có kết quả, vẽ box trắng
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), FaceRecognitionConfig.BBOX_THICKNESS)
                    frame = cv2.putText(frame, "Processing...", (x, y-10), 
                                      FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE, 
                                      (255, 255, 255), FaceRecognitionConfig.TEXT_THICKNESS)
        
        # Cập nhật danh sách tracker active
        tracker_states = active_trackers

        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), FaceRecognitionConfig.TEXT_FONT, 1, FaceRecognitionConfig.COLOR_WHITE, 2)
        
        # Display recognition statistics với thông tin tối ưu hóa
        # cv2.putText(frame, f"Processed: {total_faces_processed}", (10, 70), FaceRecognitionConfig.TEXT_FONT, 0.7, FaceRecognitionConfig.COLOR_WHITE, 2)
        # cv2.putText(frame, f"Inferences: {total_inferences_run}", (10, 110), FaceRecognitionConfig.TEXT_FONT, 0.7, (255, 255, 0), 2)
        # cv2.putText(frame, f"Cached: {cached_results}", (10, 150), FaceRecognitionConfig.TEXT_FONT, 0.7, (0, 255, 255), 2)
        # cv2.putText(frame, f"Recognized: {successful_recognitions}", (10, 190), FaceRecognitionConfig.TEXT_FONT, 0.7, FaceRecognitionConfig.COLOR_RECOGNIZED, 2)
        # cv2.putText(frame, f"Quality Rejected: {quality_rejections}", (10, 230), FaceRecognitionConfig.TEXT_FONT, 0.7, FaceRecognitionConfig.COLOR_POOR_QUALITY, 2)
        # cv2.putText(frame, f"Similarity Rejected: {similarity_rejections}", (10, 270), FaceRecognitionConfig.TEXT_FONT, 0.7, FaceRecognitionConfig.COLOR_UNKNOWN, 2)
        
        # Display threshold info và performance info
        required_confidence = (1-FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD)*100
        # cv2.putText(frame, f"Required: {required_confidence:.1f}%", (10, 310), FaceRecognitionConfig.TEXT_FONT, 0.7, (255, 255, 255), 2)
        
        # Hiển thị tỷ lệ tối ưu hóa
        if total_faces_processed > 0:
            optimization_ratio = (cached_results / (total_inferences_run + cached_results)) * 100
            # cv2.putText(frame, f"Cache Hit: {optimization_ratio:.1f}%", (10, 350), FaceRecognitionConfig.TEXT_FONT, 0.7, (0, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n📊 THỐNG KÊ NHẬN DIỆN:")
    print(f"   • Tổng số face đã xử lý: {total_faces_processed}")
    print(f"   • Số lần inference thực sự: {total_inferences_run}")
    print(f"   • Số lần dùng cache: {cached_results}")
    print(f"   • Số face nhận diện thành công: {successful_recognitions}")
    print(f"   • Số face bị từ chối do chất lượng: {quality_rejections}")
    print(f"   • Số face bị từ chối do similarity: {similarity_rejections}")
    
    if total_faces_processed > 0:
        recognition_rate = (successful_recognitions / total_faces_processed) * 100
        optimization_ratio = (cached_results / (total_inferences_run + cached_results)) * 100 if (total_inferences_run + cached_results) > 0 else 0
        print(f"   • Tỷ lệ nhận diện: {recognition_rate:.1f}%")
        print(f"   • Tỷ lệ tối ưu hóa (cache hit): {optimization_ratio:.1f}%")
        print(f"   • Giảm inference: {(1 - total_inferences_run/total_faces_processed)*100:.1f}%")
    
    logger.info("📊 FINAL STATISTICS:")
    logger.info(f"   Total faces processed: {total_faces_processed}")
    logger.info(f"   Actual inferences run: {total_inferences_run}")
    logger.info(f"   Cached results used: {cached_results}")
    logger.info(f"   Successfully recognized: {successful_recognitions}")
    logger.info(f"   Quality rejections: {quality_rejections}")
    logger.info(f"   Similarity rejections: {similarity_rejections}")
    if total_faces_processed > 0:
        optimization_ratio = (cached_results / (total_inferences_run + cached_results)) * 100 if (total_inferences_run + cached_results) > 0 else 0
        logger.info(f"   Cache hit ratio: {optimization_ratio:.1f}%")
        logger.info(f"   Inference reduction: {(1 - total_inferences_run/total_faces_processed)*100:.1f}%")
    logger.info("🏁 Face Recognition System stopped")