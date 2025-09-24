import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
from face_config import FaceRecognitionConfig

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
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FaceRecognitionConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FaceRecognitionConfig.CAMERA_HEIGHT)
    
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
            break
        
        frame_count += 1
        
        # Detect faces mới hoặc khi chưa có tracker
        if not tracker_initialized or frame_count % detection_interval == 0:
            boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
            if boxes is not None:
                # Tạo trackers mới
                trackers = []
                for box in boxes:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    bbox = (int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1]))
                    tracker.init(frame, bbox)
                    trackers.append(tracker)
                tracker_initialized = True
        
        # Update trackers và vẽ bounding boxes
        if tracker_initialized and trackers:
            for i, tracker in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:
                    # Vẽ bbox từ tracker
                    x, y, w, h = [int(v) for v in bbox]
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
                    
                    # Lấy face từ bbox để extract embedding
                    if leap % FaceRecognitionConfig.REGISTRATION_SKIP_FRAMES == 0:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            
                            face_tensor = mtcnn(face_rgb)
                            if face_tensor is not None:
                                total_tensors_processed += 1
                                
                                # Validate face quality before saving
                                is_good_quality, quality_score = validate_face_quality(face_tensor, face_rgb, mode='registration')
                                
                                if is_good_quality:
                                    good_quality_tensors += 1
                                    with torch.no_grad():
                                        embedding = model_embedding(face_tensor.unsqueeze(0).to(device))
                                        embeddings.append(embedding)
                                    count -= 1
                                    print(f"✅ Captured good quality embedding {FaceRecognitionConfig.REGISTRATION_SAMPLES-count}/{FaceRecognitionConfig.REGISTRATION_SAMPLES} (Quality: {quality_score:.2f})")
                                else:
                                    rejected_tensors += 1
                                    print(f"❌ Poor quality face rejected (Quality: {quality_score:.2f})")
        
        # Hiển thị progress và quality feedback
        cv2.putText(frame, f"Progress: {FaceRecognitionConfig.REGISTRATION_SAMPLES-count}/{FaceRecognitionConfig.REGISTRATION_SAMPLES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"User: {usr_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Good: {good_quality_tensors} | Rejected: {rejected_tensors}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Total processed: {total_tensors_processed}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Keep face steady & well-lit", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        leap += 1
        
        cv2.imshow("Face Capturing", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Xử lý kết quả
    if len(embeddings) > 0:
        embedding = torch.cat(embeddings).mean(0, keepdim=True)
        names = np.array([usr_name])
        
        # Load existing data
        existing_embeddings, existing_usernames = load_existing_data()
        
        # Merge with existing data
        if existing_embeddings is not None:
            new_embeddings = torch.cat([existing_embeddings, embedding])
            new_usernames = np.concatenate([existing_usernames, names])
        else:
            new_embeddings = embedding
            new_usernames = names
        
        # Save data
        save_data(new_embeddings, new_usernames)
        print(f'✅ Đăng ký user "{usr_name}" thành công!')
        
        # Hiển thị thống kê cuối cùng
        print("\n📊 THỐNG KÊ CHẤT LƯỢNG:")
        print(f"   • Tổng số tensor đã xử lý: {total_tensors_processed}")
        print(f"   • Số tensor đủ điều kiện: {good_quality_tensors}")
        print(f"   • Số tensor bị từ chối: {rejected_tensors}")
        if total_tensors_processed > 0:
            success_rate = (good_quality_tensors / total_tensors_processed) * 100
            print(f"   • Tỷ lệ thành công: {success_rate:.1f}%")
        
    else:
        print("❌ Không có embedding nào được capture. Đăng ký thất bại.")
        print("\n📊 THỐNG KÊ CHẤT LƯỢNG:")
        print(f"   • Tổng số tensor đã xử lý: {total_tensors_processed}")
        print(f"   • Số tensor đủ điều kiện: {good_quality_tensors}")
        print(f"   • Số tensor bị từ chối: {rejected_tensors}")
        if total_tensors_processed > 0:
            success_rate = (good_quality_tensors / total_tensors_processed) * 100
            print(f"   • Tỷ lệ thành công: {success_rate:.1f}%")

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