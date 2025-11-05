import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
from face_config import FaceRecognitionConfig
import psycopg2
from psycopg2.extras import execute_values
import time
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresFaceDatabase:
    """PostgreSQL database interface for face recognition using pgvector"""
    
    def __init__(self, connection_params):
        self.conn = psycopg2.connect(
            host=connection_params.get('host', 'localhost'),
            port=connection_params.get('port', 5433),
            dbname=connection_params.get('dbname', 'healthmed'),
            user=connection_params.get('user', 'paperless'),
            password=connection_params.get('password', 'paperless')
        )
        logger.info("✅ Connected to PostgreSQL database")
    
    def vec2pgvector(self, embedding):
        """Convert numpy/torch array to PostgreSQL vector format"""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        # Format: [0.1,0.2,0.3,...]
        return '[' + ','.join(map(str, embedding)) + ']'
    
    def get_recognition(self, embedding, threshold=0.4):
        """Query database for similar face using pgvector cosine distance"""
        try:
            with self.conn.cursor() as cur:
                embedding_str = self.vec2pgvector(embedding)
                
                query = """
                    SELECT id, name, distance
                    FROM (
                        SELECT id, name, embedding <=> %s::vector AS distance
                        FROM Person
                    ) AS sub
                    WHERE distance < %s
                    ORDER BY distance
                    LIMIT 1;
                """
                
                cur.execute(query, (embedding_str, threshold))
                result = cur.fetchone()
                
                if result:
                    person_id, name, distance = result
                    confidence = 1 - distance  # Convert distance to confidence
                    return {
                        'id': person_id,
                        'name': name,
                        'confidence': confidence,
                        'distance': distance
                    }
                else:
                    return None
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            return None
    
    def close(self):
        self.conn.close()


class LiveFaceRecognizer:
    """Live face recognition using webcam/video and PostgreSQL"""
    
    def __init__(self, db_config, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load face recognition model
        logger.info("Loading face recognition model...")
        self.model = InceptionResnetV1(
            classify=False,
            pretrained="casia-webface"
        ).to(self.device)
        self.model.eval()
        
        # Initialize MTCNN
        logger.info("Initializing MTCNN...")
        self.mtcnn = MTCNN(
            margin=FaceRecognitionConfig.MTCNN_MARGIN,
            keep_all=FaceRecognitionConfig.MTCNN_KEEP_ALL,
            select_largest=FaceRecognitionConfig.MTCNN_SELECT_LARGEST,
            post_process=FaceRecognitionConfig.MTCNN_POST_PROCESS,
            device=self.device
        )
        
        # Initialize database
        logger.info("Initializing database connection...")
        self.db = PostgresFaceDatabase(db_config)
        
        # Recognition threshold
        self.recognition_threshold = 0.3  # Cosine distance threshold
        self.quality_check = False
        
        logger.info("✅ Face recognizer initialized")
    
    def validate_face_quality(self, face_tensor, frame_roi, mode='inference'):
        """Check face quality before recognition"""
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
        
        is_good_quality = quality_score >= thresholds['quality_threshold']
        
        return is_good_quality, quality_score
    
    def recognize_face(self, frame):
        """Process frame and recognize faces"""
        # Detect faces with MTCNN
        try:
            detection_result = self.mtcnn.detect(frame)
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return [], frame
        
        if detection_result is None:
            return [], frame
        
        # Handle different return formats
        if isinstance(detection_result, tuple):
            boxes = detection_result[0]
        else:
            boxes = detection_result
        
        if boxes is None or len(boxes) == 0:
            return [], frame
        
        logger.debug(f"Detected {len(boxes)} face(s)")
        
        results = []
        processed_frame = frame.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Extract face region
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            # Convert to RGB for MTCNN
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            try:
                # Get face tensor from MTCNN (extracts and aligns face)
                face_tensor = self.mtcnn(face_rgb)
                
                if face_tensor is None:
                    continue
                
                # Validate face quality
                if self.quality_check:
                    is_good_quality, quality_score = self.validate_face_quality(
                        face_tensor, face_rgb, mode='inference'
                    )
                    
                    if not is_good_quality:
                        # Draw poor quality
                        color = FaceRecognitionConfig.COLOR_POOR_QUALITY
                        text = f"Poor Quality ({quality_score*100:.1f})"
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, FaceRecognitionConfig.BBOX_THICKNESS)
                        cv2.putText(processed_frame, text, (x1, y1-10),
                                FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE,
                                color, FaceRecognitionConfig.TEXT_THICKNESS)
                        continue
                
                # Generate embedding
                with torch.no_grad():
                    detect_embeds = self.model(face_tensor.unsqueeze(0).to(self.device))
                
                # Query database for recognition
                person = self.db.get_recognition(detect_embeds.squeeze(0), self.recognition_threshold)
                
                if person:
                    # Person recognized
                    name = person['name']
                    confidence = person['confidence']
                    
                    # Draw bounding box and label
                    color = FaceRecognitionConfig.COLOR_RECOGNIZED
                    text = f"{name} ({confidence*100:.1f}%)"
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, FaceRecognitionConfig.BBOX_THICKNESS)
                    cv2.putText(processed_frame, text, (x1, y1-10),
                              FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE,
                              color, FaceRecognitionConfig.TEXT_THICKNESS)
                    
                    results.append({
                        'name': name,
                        'confidence': confidence,
                        'status': 'recognized',
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    logger.info(f"Recognized: {name} ({confidence*100:.1f}%)")
                else:
                    # Unknown person
                    color = FaceRecognitionConfig.COLOR_UNKNOWN
                    text = "Unknown"
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, FaceRecognitionConfig.BBOX_THICKNESS)
                    cv2.putText(processed_frame, text, (x1, y1-10),
                              FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE,
                              color, FaceRecognitionConfig.TEXT_THICKNESS)
                    
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'status': 'unknown',
                        'bbox': (x1, y1, x2, y2)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing face: {e}")
                continue
        
        return results, processed_frame
    
    def run(self, source=None, fps_display=True, headless=False):
        """Run live face recognition on webcam or video file"""
        if source is None:
            source = 0  # Default to webcam
        
        # Check if display is available - disable display by default
        display_available = False
        logger.info("Running in headless mode - no display window will be shown")
        
        # Open video capture - handle both int (webcam index) and str (video file path)
        cap = cv2.VideoCapture(int(source) if isinstance(source, str) and source.isdigit() else source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return
        
        logger.info(f"✅ Video source opened: {source}")
        
        # Set some properties for webcam
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info("Set webcam properties")
        
        frame_count = 0
        total_time = 0.0
        prev_time = time.time()
        
        logger.info("Starting frame capture loop...")
        while True:
            if frame_count == 0:
                logger.info("About to read first frame...")
            elif frame_count == 1:
                logger.info("About to read second frame...")
            
            ret, frame = cap.read()
            
            if frame_count == 0:
                logger.info("First frame read completed")
            elif frame_count == 1:
                logger.info("Second frame read completed")
            
            if not ret:
                logger.info("End of video or failed to capture frame")
                break
            
            if frame_count == 0:
                logger.info(f"First frame captured successfully: {frame.shape}")
                logger.info("Processing first frame (this may take a moment)...")
            elif frame_count == 1:
                logger.info(f"Second frame captured: {frame.shape}")
            
            # Process frame
            start_time = time.time()
            if frame_count == 0:
                logger.info("Calling recognize_face...")
            results, processed_frame = self.recognize_face(frame)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if frame_count == 0:
                logger.info(f"First frame processed in {processing_time:.2f}s")
                logger.info("Calculating FPS...")
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            if frame_count == 0:
                logger.info(f"FPS calculated: {fps:.2f}")
            
            # Add FPS display if requested
            if fps_display:
                cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                          (10, 30), FaceRecognitionConfig.TEXT_FONT, 
                          0.7, FaceRecognitionConfig.COLOR_WHITE, 2)
            
            if frame_count == 0:
                logger.info("Displaying first frame..." if display_available else "Skipping display (headless mode)...")
            
            # Display frame
            if display_available:
                try:
                    cv2.imshow("Face Recognition - Live", processed_frame)
                except cv2.error as e:
                    logger.error(f"cv2.imshow error: {e}")
                    display_available = False
            
            if frame_count == 0:
                logger.info("Frame displayed (or skipped), waiting for key press..." if display_available else "Skipping key check (headless mode)")
            
            # Print results
            if results:
                for result in results:
                    logger.info(f"Face {frame_count}: {result['name']} "
                              f"(confidence: {result['confidence']:.2f}, "
                              f"status: {result['status']})")
            
            # Check for exit
            if display_available:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Exiting...")
                    break
            
            # In headless mode, allow interruption with Ctrl+C or limit frames
            if not display_available and frame_count >= 1000:  # Limit to 100 frames in headless mode
                logger.info("Reached frame limit in headless mode")
                break
            
            if frame_count == 0:
                logger.info("Starting continuous loop...")
            
            frame_count += 1
        
        cap.release()
        if display_available:
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames")
        logger.info(f"Average latency: {total_time / frame_count * 1000:.2f} ms")
        logger.info(f"Average FPS: {frame_count / total_time:.2f}")
    
    def close(self):
        """Clean up resources"""
        self.db.close()


def main():
    # Parse command line arguments
    source = 0  # Default to webcam
    headless = False
    
    if len(sys.argv) > 1:
        # Check for headless flag
        if '--headless' in sys.argv:
            headless = True
            sys.argv.remove('--headless')
        elif '-h' in sys.argv:
            headless = True
            sys.argv.remove('-h')
        
        if len(sys.argv) > 1:
            source = sys.argv[1]
            # Try to convert to int if it's a string
            if isinstance(source, str) and source.isdigit():
                source = int(source)
    
    # Database configuration
    db_config = {
        'dbname': 'healthmed',
        'user': 'paperless',
        'password': 'paperless',
        'host': 'localhost',
        'port': '5433'
    }
    
    # Create recognizer
    recognizer = LiveFaceRecognizer(db_config, 'cpu')
    
    try:
        # Run recognition
        recognizer.run(source, fps_display=True, headless=headless)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()

