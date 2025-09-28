import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
from face_config import FaceRecognitionConfig
import argparse
from pathlib import Path
from PIL import Image

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
        embeddings = torch.load(embedding_file, map_location=device)
        print(f"Loaded embeddings: {embeddings.shape}")
    
    # Load usernames
    username_file = os.path.join(DATA_PATH, "usernames.npy")
    if os.path.exists(username_file):
        usernames = np.load(username_file, allow_pickle=True)
        print(f"Loaded usernames: {usernames}")
    
    return embeddings, usernames

def validate_face_quality(face_tensor, frame_roi, mode='registration'):
    """Kiểm tra chất lượng face trước khi lưu embedding"""
    thresholds = FaceRecognitionConfig.get_quality_thresholds(mode)
    
    quality_score = 0.0
    checks_passed = 0
    total_checks = 4
    quality_details = {}
    
    # Convert tensor to numpy for OpenCV operations
    if isinstance(face_tensor, torch.Tensor):
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_np = ((face_np + 1) * 127.5).astype(np.uint8)
    else:
        face_np = frame_roi
    
    # 1. Blur Detection
    gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY) if len(face_np.shape) == 3 else face_np
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_pass = blur_score > thresholds['blur_threshold']
    quality_details['blur_score'] = blur_score
    
    if blur_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # 2. Face Size Validation
    face_height, face_width = gray.shape[:2]
    size_pass = face_height >= thresholds['min_face_size'] and face_width >= thresholds['min_face_size']
    
    if size_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # 3. Lighting Condition
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[200:])
    total_pixels = face_height * face_width
    
    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels
    lighting_pass = dark_ratio < thresholds['dark_ratio_threshold'] and bright_ratio < thresholds['bright_ratio_threshold']
    
    if lighting_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # 4. Pose Estimation
    mid_point = face_width // 2
    left_half = gray[:, :mid_point]
    right_half = gray[:, mid_point:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_resized = cv2.resize(left_half, (min_width, face_height))
    right_resized = cv2.resize(right_half_flipped, (min_width, face_height))
    
    mse = np.mean((left_resized.astype(float) - right_resized.astype(float)) ** 2)
    pose_pass = mse < thresholds['pose_threshold']
    
    if pose_pass:
        quality_score += 0.25
        checks_passed += 1
    
    # Print quality details
    print(f"📊 QUALITY CHECK RESULTS:")
    print(f"   • Blur Score: {blur_score:.1f} {'✅' if blur_pass else '❌'} (threshold: {thresholds['blur_threshold']})")
    print(f"   • Face Size: {face_width}x{face_height} {'✅' if size_pass else '❌'} (min: {thresholds['min_face_size']})")
    print(f"   • Lighting - Dark: {dark_ratio:.3f} {'✅' if dark_ratio < thresholds['dark_ratio_threshold'] else '❌'}")
    print(f"   • Lighting - Bright: {bright_ratio:.3f} {'✅' if bright_ratio < thresholds['bright_ratio_threshold'] else '❌'}")
    print(f"   • Pose MSE: {mse:.1f} {'✅' if pose_pass else '❌'} (threshold: {thresholds['pose_threshold']})")
    print(f"   • Checks passed: {checks_passed}/{total_checks}")
    print(f"   • Overall Quality Score: {quality_score:.2f}")
    
    is_good_quality = quality_score >= thresholds['quality_threshold']
    
    return is_good_quality, quality_score, quality_details

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

def register_user_from_image(image_path, username=None):
    """Register a new user from a single image"""
    
    # Validate image path
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file ảnh: {image_path}")
        return False
    
    # Get username if not provided
    if username is None:
        username = input("Nhập tên user: ").strip()
        if not username:
            print("❌ Tên không được để trống")
            return False
    
    print(f"🖼️  Đang xử lý ảnh: {image_path}")
    print(f"👤 User: {username}")
    
    # Load image
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Không thể đọc file ảnh: {image_path}")
            return False
        
        print(f"✅ Đã load ảnh thành công: {frame.shape}")
        
    except Exception as e:
        print(f"❌ Lỗi khi load ảnh: {e}")
        return False
    
    # Initialize models
    print("🔧 Khởi tạo models...")
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
    
    # Convert BGR to RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image for MTCNN compatibility
    frame_pil = Image.fromarray(frame_rgb)

    # Detect faces
    print("🔍 Detecting faces...")
    boxes, probs, points_list = mtcnn.detect(frame_pil, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        print("❌ Không phát hiện được khuôn mặt nào trong ảnh")
        print("💡 Đảm bảo:")
        print("   - Ảnh có chứa khuôn mặt rõ ràng")
        print("   - Khuôn mặt không bị che khuất")
        print("   - Ảnh có độ phân giải đủ cao")
        return False
    
    print(f"✅ Phát hiện được {len(boxes)} khuôn mặt")
    
    # Process each detected face
    embeddings_list = []
    face_processed = False
    
    for i, (box, prob, landmarks) in enumerate(zip(boxes, probs, points_list)):
        print(f"\n🔬 Xử lý khuôn mặt {i+1}/{len(boxes)} (confidence: {prob:.3f})")
        
        # Skip low confidence detections
        if prob < 0.9:
            print(f"⚠️  Bỏ qua khuôn mặt {i+1} do confidence thấp: {prob:.3f}")
            continue
        
        # Extract face region for quality check
        x1, y1, x2, y2 = [int(coord) for coord in box]
        face_roi = frame_rgb[y1:y2, x1:x2]
        
        # Get face tensor from MTCNN
        try:
            face_tensor = mtcnn(face_roi)
            if face_tensor is None:
                print(f"❌ MTCNN không thể extract face {i+1}")
                continue
                
        except Exception as e:
            print(f"❌ Lỗi khi extract face {i+1}: {e}")
            continue
        
        # Validate face quality
        print(f"🔍 Kiểm tra chất lượng khuôn mặt {i+1}...")
        is_good_quality, quality_score, quality_details = validate_face_quality(
            face_tensor, face_roi, mode='registration'
        )
        
        if not is_good_quality:
            print(f"❌ Khuôn mặt {i+1} không đạt chất lượng yêu cầu (score: {quality_score:.2f})")
            
            # Ask user if they want to proceed anyway
            choice = input("Bạn có muốn tiếp tục với khuôn mặt này không? (y/n): ").lower().strip()
            if choice not in ['y', 'yes', 'có']:
                print(f"⏭️  Bỏ qua khuôn mặt {i+1}")
                continue
        else:
            print(f"✅ Khuôn mặt {i+1} đạt chất lượng tốt (score: {quality_score:.2f})")
        
        # Generate embedding
        print(f"🧠 Tạo embedding cho khuôn mặt {i+1}...")
        try:
            with torch.no_grad():
                embedding = model_embedding(face_tensor.unsqueeze(0).to(device))
                embeddings_list.append(embedding)
                face_processed = True
                print(f"✅ Đã tạo embedding thành công cho khuôn mặt {i+1}")
                
                # If we only want to use the best/first face, break here
                if FaceRecognitionConfig.MTCNN_SELECT_LARGEST:
                    break
                    
        except Exception as e:
            print(f"❌ Lỗi khi tạo embedding cho khuôn mặt {i+1}: {e}")
            continue
    
    # Check if any face was processed successfully
    if not face_processed or len(embeddings_list) == 0:
        print("❌ Không có khuôn mặt nào được xử lý thành công")
        return False
    
    # Average embeddings if multiple faces were processed
    if len(embeddings_list) > 1:
        print(f"🔢 Tính trung bình từ {len(embeddings_list)} embeddings...")
        final_embedding = torch.cat(embeddings_list).mean(0, keepdim=True)
    else:
        final_embedding = embeddings_list[0]
    
    print(f"✅ Embedding cuối cùng: {final_embedding.shape}")
    
    # Load existing data and merge
    print("💾 Lưu dữ liệu...")
    existing_embeddings, existing_usernames = load_existing_data()
    
    # Check if user already exists
    if existing_usernames is not None:
        existing_usernames_list = existing_usernames.tolist() if isinstance(existing_usernames, np.ndarray) else existing_usernames
        if username in existing_usernames_list:
            choice = input(f"⚠️  User '{username}' đã tồn tại. Bạn có muốn ghi đè? (y/n): ").lower().strip()
            if choice not in ['y', 'yes', 'có']:
                print("❌ Hủy đăng ký user")
                return False
            
            # Remove existing user
            user_index = existing_usernames_list.index(username)
            existing_usernames = np.delete(existing_usernames, user_index)
            if existing_embeddings is not None:
                existing_embeddings = torch.cat([existing_embeddings[:user_index], existing_embeddings[user_index+1:]])
    
    # Merge with existing data
    names = np.array([username])
    if existing_embeddings is not None and len(existing_embeddings) > 0:
        new_embeddings = torch.cat([existing_embeddings, final_embedding])
        new_usernames = np.concatenate([existing_usernames, names])
    else:
        new_embeddings = final_embedding
        new_usernames = names
    
    # Save data
    try:
        save_data(new_embeddings, new_usernames)
        print(f'🎉 Đăng ký user "{username}" thành công!')
        print(f"📊 Tổng số user trong hệ thống: {len(new_usernames)}")
        
        # Show updated user list
        show_user_list(new_usernames)
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu dữ liệu: {e}")
        return False

def main_menu():
    """Main menu system"""
    while True:
        print("\n" + "="*50)
        print("🎭 FACE REGISTRATION FROM IMAGE")
        print("="*50)
        print("1. 🖼️  Đăng ký user từ ảnh")
        print("2. 📋 Xem danh sách user")
        print("3. 🗑️  Xóa user")
        print("4. ⚙️  Hiển thị config")
        print("5. ❌ Thoát")
        print("="*50)
        
        choice = input("Chọn chức năng (1-5): ").strip()
        
        if choice == '1':
            image_path = input("Nhập đường dẫn ảnh: ").strip().strip('"\'')
            if image_path:
                register_user_from_image(image_path)
            else:
                print("❌ Đường dẫn ảnh không được để trống")
                
        elif choice == '2':
            _, usernames = load_existing_data()
            show_user_list(usernames)
            
        elif choice == '3':
            embeddings, usernames = load_existing_data()
            delete_user(embeddings, usernames)
            
        elif choice == '4':
            FaceRecognitionConfig.print_config()
            
        elif choice == '5':
            print("👋 Tạm biệt!")
            break
            
        else:
            print("❌ Vui lòng chọn từ 1-5")

def main():
    """Main function with command line support"""
    parser = argparse.ArgumentParser(description='Face Registration from Single Image')
    parser.add_argument('--image', '-i', type=str, help='Path to image file')
    parser.add_argument('--name', '-n', type=str, help='User name')
    parser.add_argument('--batch', '-b', type=str, help='Path to directory containing images (filename = username)')
    
    args = parser.parse_args()
    
    # Command line mode
    if args.image:
        if register_user_from_image(args.image, args.name):
            print("✅ Command line registration successful")
        else:
            print("❌ Command line registration failed")
            return
    
    # Batch processing mode
    elif args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"❌ Batch directory not found: {batch_dir}")
            return
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(batch_dir.glob(f'*{ext}'))
            image_files.extend(batch_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No image files found in: {batch_dir}")
            return
            
        print(f"🔍 Found {len(image_files)} image files")
        
        success_count = 0
        for image_file in image_files:
            username = image_file.stem  # filename without extension
            print(f"\n📸 Processing: {image_file.name} -> {username}")
            
            if register_user_from_image(str(image_file), username):
                success_count += 1
            else:
                print(f"❌ Failed to process: {image_file.name}")
        
        print(f"\n🎯 Batch processing completed: {success_count}/{len(image_files)} successful")
    
    # Interactive menu mode
    else:
        main_menu()

if __name__ == "__main__":
    main()