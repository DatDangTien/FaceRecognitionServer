import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
import logging
import datetime
import threading
from face_config import FaceRecognitionConfig

# Configure OpenCV to avoid Qt5 threading issues
# cv2.setUseOptimized(True)

# Try to set Qt5 to use single-threaded mode to avoid threading conflicts
# try:
#     import os
#     # Use xcb platform which is more stable for remote desktop connections
#     if 'DISPLAY' in os.environ:
#         os.environ['QT_QPA_PLATFORM'] = 'xcb'
#     else:
#         os.environ['QT_QPA_PLATFORM'] = 'minimal'
# except:
#     pass

# print(f"os.environ: {os.environ['QT_QPA_PLATFORM']}")

# Check for headless mode
# HEADLESS_MODE = os.environ.get('HEADLESS', 'false').lower() == 'true'
HEADLESS_MODE = True

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'face_recognition_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

DATA_PATH = './data'
os.makedirs(DATA_PATH, exist_ok=True)

def load_existing_data():
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
        print(f"Loaded embeddings: {embeddings.shape}")
    
    # Load usernames
    username_file = os.path.join(DATA_PATH, "usernames.npy")
    if os.path.exists(username_file):
        usernames = np.load(username_file)
        print(f"Loaded usernames: {usernames}")
    
    return embeddings, usernames

def validate_face_quality(face_tensor, frame_roi, mode='registration'):
    """Kiểm tra chất lượng face trước khi lưu embedding"""
    # Get thresholds from config based on mode
    thresholds = FaceRecognitionConfig.get_quality_thresholds(mode)
    
    quality_score = 0.0
    checks_passed = 0
    total_checks = 4
    
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
    if blur_score > thresholds['blur_threshold']:
        quality_score += 0.25
        checks_passed += 1
    
    # 2. Face Size Validation
    face_height, face_width = gray.shape[:2]
    if face_height >= thresholds['min_face_size'] and face_width >= thresholds['min_face_size']:
        quality_score += 0.25
        checks_passed += 1
    
    # 3. Lighting Condition (histogram analysis)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[200:])
    total_pixels = face_height * face_width
    
    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels
    
    if dark_ratio < thresholds['dark_ratio_threshold'] and bright_ratio < thresholds['bright_ratio_threshold']:
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
    if mse < thresholds['pose_threshold']:
        quality_score += 0.25
        checks_passed += 1
    
    # Print quality details for debugging
    print(f"Quality Check - Blur: {blur_score:.1f}, Size: {face_width}x{face_height}, "
          f"Dark: {dark_ratio:.2f}, Bright: {bright_ratio:.2f}, Pose MSE: {mse:.1f}")
    print(f"Checks passed: {checks_passed}/{total_checks}, Quality score: {quality_score:.2f}")
    
    return quality_score >= thresholds['quality_threshold'], quality_score

def save_data(embeddings, usernames):
    """Save embeddings and usernames to files"""
    # Save embeddings
    if device.type == 'cpu':
        torch.save(embeddings, os.path.join(DATA_PATH, "faceslistCPU.pth"))
    else:
        torch.save(embeddings, os.path.join(DATA_PATH, "faceslist.pth"))
    
    # Save usernames
    np.save(os.path.join(DATA_PATH, "usernames.npy"), usernames)
    print(f"✅ Data saved successfully!")

def show_user_list(usernames):
    """Display list of registered users"""
    if usernames is None or len(usernames) == 0:
        print("📋 Danh sách user trống")
        return
    
    print("\n📋 DANH SÁCH USER ĐÃ ĐĂNG KÝ:")
    print("-" * 40)
    for i, username in enumerate(usernames, 1):
        print(f"{i}. {username}")
    print("-" * 40)
    print(f"Tổng cộng: {len(usernames)} user(s)")

def delete_user(embeddings, usernames):
    """Delete a specific user"""
    if usernames is None or len(usernames) == 0:
        print("❌ Không có user nào để xóa")
        return embeddings, usernames
    
    show_user_list(usernames)
    
    while True:
        try:
            choice = input(f"\nChọn user cần xóa (1-{len(usernames)}) hoặc 'q' để hủy: ").strip()
            if choice.lower() == 'q':
                print("❌ Hủy xóa user")
                return embeddings, usernames
            
            user_index = int(choice) - 1
            if 0 <= user_index < len(usernames):
                user_to_delete = usernames[user_index]
                confirm = input(f"Bạn có chắc muốn xóa user '{user_to_delete}'? (y/n): ").lower().strip()
                
                if confirm in ['y', 'yes', 'có']:
                    # Remove user from usernames
                    new_usernames = np.delete(usernames, user_index)
                    
                    # Remove corresponding embedding
                    if embeddings is not None:
                        new_embeddings = torch.cat([embeddings[:user_index], embeddings[user_index+1:]])
                    else:
                        new_embeddings = None
                    
                    # Save updated data
                    save_data(new_embeddings, new_usernames)
                    print(f"✅ Đã xóa user '{user_to_delete}' thành công!")
                    return new_embeddings, new_usernames
                else:
                    print("❌ Hủy xóa user")
                    return embeddings, usernames
            else:
                print(f"❌ Vui lòng chọn số từ 1 đến {len(usernames)}")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ")

def register_new_user():
    """Register a new user"""
    usr_name = input("Input your name: ")
    if not usr_name.strip():
        print("❌ Tên không được để trống")
        return
    
    count = FaceRecognitionConfig.REGISTRATION_SAMPLES
    embeddings = []
    
    # Biến đếm để theo dõi quality
    total_tensors_processed = 0
    good_quality_tensors = 0
    rejected_tensors = 0
    
    mtcnn = MTCNN(
        margin=FaceRecognitionConfig.MTCNN_MARGIN,
        keep_all=FaceRecognitionConfig.MTCNN_KEEP_ALL,
        select_largest=FaceRecognitionConfig.MTCNN_SELECT_LARGEST,
        post_process=FaceRecognitionConfig.MTCNN_POST_PROCESS,
        device=device
    )
    
    model_embedding = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model_embedding.eval()
    # print(f"model_embedding: {model_embedding}")
    
    logging.info("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        return
        
    # Set camera properties
    width_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, FaceRecognitionConfig.CAMERA_WIDTH)
    height_success = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FaceRecognitionConfig.CAMERA_HEIGHT)
    
    if not width_success or not height_success:
        logging.warning("Failed to set camera resolution")
    
    # Verify camera settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logging.info(f"Camera initialized with resolution: {actual_width}x{actual_height}")
    
    # Test camera read
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        logging.error("Failed to read test frame from camera")
        cap.release()
        return
    logging.info("Camera test read successful")
    
    # Initialize display window outside the loop (only if not in headless mode)
    display_available = False
    if not HEADLESS_MODE:
        try:
            # Try to create window with error handling
            cv2.namedWindow("Face Capturing", cv2.WINDOW_NORMAL)
            logging.info("Display window created successfully")
            
            # Test display with a simple image first
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("Face Capturing", test_img)
            cv2.waitKey(0)  # Test display for 1ms
            
            # If successful, try with the actual frame
            cv2.imshow("Face Capturing", test_frame)
            cv2.waitKey(0)
            
            display_available = True
            logging.info("Display window created successfully")
        except Exception as e:
            logging.error(f"Failed to create display window: {str(e)}")
            logging.warning("Continuing without display window...")
            display_available = False
            # Try to clean up any partial window creation
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    # Output image dir
    output_image_dir = "logs/img"
    os.makedirs(output_image_dir, exist_ok=True)
    
    # Khởi tạo tracker
    trackers = []
    tracker_initialized = False
    detection_interval = FaceRecognitionConfig.DETECTION_INTERVAL
    frame_count = 0
    
    leap = 1
    print(f"🎥 Bắt đầu capture face cho user '{usr_name}'...")
    print("Nhấn ESC để thoát")
    
    # Hiển thị config hiện tại
    FaceRecognitionConfig.print_config()
    
    while cap.isOpened() and count > 0:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from camera")
            break
        
        frame_count += 1
        logging.debug(f"Processing frame {frame_count}")
        
        # Detect faces mới hoặc khi chưa có tracker
        if not tracker_initialized or frame_count % detection_interval == 0:
            logging.debug(f"Running face detection (tracker_initialized={tracker_initialized}, frame_count={frame_count})")
            boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
            
            if boxes is not None:
                logging.info(f"Detected {len(boxes)} faces in frame {frame_count}")
                # Tạo trackers mới
                trackers = []
                for i, box in enumerate(boxes):
                    tracker = cv2.legacy.TrackerCSRT_create()
                    bbox = (int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1]))
                    tracker.init(frame, bbox)
                    trackers.append(tracker)
                    logging.debug(f"Initialized tracker {i+1} at bbox {bbox}")
                tracker_initialized = True
            else:
                logging.warning(f"No faces detected in frame {frame_count}")
        
        # Update trackers và vẽ bounding boxes
        if tracker_initialized and trackers:
            logging.debug(f"Updating {len(trackers)} trackers")
            for i, tracker in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:
                    # Vẽ bbox từ tracker
                    x, y, w, h = [int(v) for v in bbox]
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
                    
                    # Lấy face từ bbox để extract embedding
                    if leap % FaceRecognitionConfig.REGISTRATION_SKIP_FRAMES == 0:
                        logging.debug(f"Processing face ROI at frame {frame_count} (leap={leap})")
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            logging.debug(f"Face ROI extracted: {face_roi.shape}")
                            cv2.imwrite(os.path.join(output_image_dir, f"face_roi_{frame_count}.jpg"), face_roi)
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            
                            try:
                                face_tensor = mtcnn(face_rgb)
                            except Exception as e:
                                logging.error(f"MTCNN error: {e}")
                            if face_tensor is not None:
                                total_tensors_processed += 1
                                logging.info(f"Face tensor generated successfully: {face_tensor.shape}")
                                
                                # Validate face quality before saving
                                logging.debug("Starting face quality validation")
                                is_good_quality, quality_score = validate_face_quality(face_tensor, face_rgb, mode='registration')
                                
                                if is_good_quality:
                                    good_quality_tensors += 1
                                    logging.info(f"Face passed quality check with score: {quality_score:.2f}")
                                    
                                    logging.debug("Generating face embedding")
                                    with torch.no_grad():
                                        embedding = model_embedding(face_tensor.unsqueeze(0).to(device))
                                        embeddings.append(embedding)
                                        logging.info(f"Face embedding generated successfully: {embedding.shape}")
                                    
                                    count -= 1
                                    print(f"✅ Captured good quality embedding {FaceRecognitionConfig.REGISTRATION_SAMPLES-count}/{FaceRecognitionConfig.REGISTRATION_SAMPLES} (Quality: {quality_score:.2f})")
                                    logging.info(f"Progress: {FaceRecognitionConfig.REGISTRATION_SAMPLES-count}/{FaceRecognitionConfig.REGISTRATION_SAMPLES} embeddings captured")
                                else:
                                    rejected_tensors += 1
                                    logging.warning(f"Face rejected due to poor quality. Score: {quality_score:.2f}")
                                    print(f"❌ Poor quality face rejected (Quality: {quality_score:.2f})")
                            else:
                                logging.warning("MTCNN failed to generate face tensor")
                        else:
                            logging.warning("Empty face ROI detected")
        
        # Hiển thị progress và quality feedback
        cv2.putText(frame, f"Progress: {FaceRecognitionConfig.REGISTRATION_SAMPLES-count}/{FaceRecognitionConfig.REGISTRATION_SAMPLES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"User: {usr_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Good: {good_quality_tensors} | Rejected: {rejected_tensors}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Total processed: {total_tensors_processed}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Keep face steady & well-lit", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        leap += 1
        
        # Display frame if window was successfully created outside the loop
        if tracker_initialized and display_available:
            try:
                logging.debug("Displaying frame")
                cv2.imwrite(os.path.join(output_image_dir, f"frame_{frame_count}.jpg"), frame)
                
                if not HEADLESS_MODE:
                    cv2.imshow("Face Capturing", frame)
                
                # Use a shorter waitKey timeout to prevent hanging
                key = cv2.waitKey(1)
                
                if key & 0xFF == 27:
                    logging.info("ESC key pressed, exiting capture loop")
                    break
                elif key == -1:  # No key pressed
                    pass
                else:
                    logging.debug(f"Key pressed: {key}")
                    
            except Exception as e:
                logging.error(f"Error displaying frame: {str(e)}")
                logging.warning("Continuing without display...")
                display_available = False  # Disable display for remaining frames
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Xử lý kết quả
    logging.info(f"Face capture completed for user '{usr_name}'")
    logging.info(f"Total embeddings captured: {len(embeddings)}")
    
    if len(embeddings) > 0:
        logging.info("Processing captured embeddings")
        embedding = torch.cat(embeddings).mean(0, keepdim=True)
        names = np.array([usr_name])
        logging.debug(f"Final embedding shape: {embedding.shape}")
        
        # Load existing data
        logging.info("Loading existing face recognition data")
        existing_embeddings, existing_usernames = load_existing_data()
        
        # Merge with existing data
        if existing_embeddings is not None:
            logging.info(f"Merging with existing data (current users: {len(existing_usernames)})")
            new_embeddings = torch.cat([existing_embeddings, embedding])
            new_usernames = np.concatenate([existing_usernames, names])
            logging.debug(f"Updated embeddings shape: {new_embeddings.shape}")
        else:
            logging.info("No existing data found, creating new dataset")
            new_embeddings = embedding
            new_usernames = names
        
        # Save data
        logging.info("Saving updated face recognition data")
        save_data(new_embeddings, new_usernames)
        print(f'✅ Đăng ký user "{usr_name}" thành công!')
        logging.info(f'Successfully registered user "{usr_name}"')
        
        # Hiển thị thống kê cuối cùng
        print("\n📊 THỐNG KÊ CHẤT LƯỢNG:")
        print(f"   • Tổng số tensor đã xử lý: {total_tensors_processed}")
        print(f"   • Số tensor đủ điều kiện: {good_quality_tensors}")
        print(f"   • Số tensor bị từ chối: {rejected_tensors}")
        
        # Log detailed statistics
        logging.info("Final Quality Statistics:")
        logging.info(f"Total tensors processed: {total_tensors_processed}")
        logging.info(f"Good quality tensors: {good_quality_tensors}")
        logging.info(f"Rejected tensors: {rejected_tensors}")
        
        if total_tensors_processed > 0:
            success_rate = (good_quality_tensors / total_tensors_processed) * 100
            print(f"   • Tỷ lệ thành công: {success_rate:.1f}%")
            logging.info(f"Success rate: {success_rate:.1f}%")
        
    else:
        error_msg = "No embeddings captured. Registration failed."
        logging.error(error_msg)
        print("❌ Không có embedding nào được capture. Đăng ký thất bại.")
        
        print("\n📊 THỐNG KÊ CHẤT LƯỢNG:")
        print(f"   • Tổng số tensor đã xử lý: {total_tensors_processed}")
        print(f"   • Số tensor đủ điều kiện: {good_quality_tensors}")
        print(f"   • Số tensor bị từ chối: {rejected_tensors}")
        
        # Log failure statistics
        logging.error("Registration Failed - Final Statistics:")
        logging.error(f"Total tensors processed: {total_tensors_processed}")
        logging.error(f"Good quality tensors: {good_quality_tensors}")
        logging.error(f"Rejected tensors: {rejected_tensors}")
        
        if total_tensors_processed > 0:
            success_rate = (good_quality_tensors / total_tensors_processed) * 100
            print(f"   • Tỷ lệ thành công: {success_rate:.1f}%")
            logging.error(f"Success rate: {success_rate:.1f}%")

def main_menu():
    """Main menu system"""
    while True:
        print("\n" + "="*50)
        print("🎭 FACE RECOGNITION SYSTEM")
        print("="*50)
        print("1. 📝 Đăng ký user mới")
        print("2. 📋 Xem danh sách user")
        print("3. 🗑️  Xóa user")
        print("4. ❌ Thoát")
        print("="*50)
        
        choice = input("Chọn chức năng (1-4): ").strip()
        
        if choice == '1':
            register_new_user()
        elif choice == '2':
            _, usernames = load_existing_data()
            show_user_list(usernames)
        elif choice == '3':
            embeddings, usernames = load_existing_data()
            delete_user(embeddings, usernames)
        elif choice == '4':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Vui lòng chọn từ 1-4")

if __name__ == "__main__":
    main_menu()