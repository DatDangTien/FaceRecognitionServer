import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import logging
from face_config import FaceRecognitionConfig
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition_websocket.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_tracker():
    """Create a tracker compatible with different OpenCV versions"""
    try:
        # Try OpenCV 4.5.1+ format
        return cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            # Try legacy format for OpenCV 4.0+
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            try:
                # Try older format
                return cv2.TrackerCSRT.create()
            except AttributeError:
                try:
                    # Try KCF tracker (legacy)
                    return cv2.legacy.TrackerKCF_create()
                except AttributeError:
                    try:
                        # Try older KCF format
                        return cv2.TrackerKCF.create()
                    except AttributeError:
                        try:
                            # Try even older format
                            return cv2.cv2.TrackerKCF_create()
                        except:
                            logger.error("No compatible tracker found. Using simple bounding box.")
                            return None

class FaceRecognitionWebSocketServer:
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Test tracker creation
        test_tracker = create_tracker()
        logger.info(f"Tracker type: {type(test_tracker).__name__ if test_tracker is not None else 'None'}")
        
        if test_tracker is None:
            logger.warning("⚠️  No OpenCV tracker available. Using detection-only mode.")
            self.use_tracking = False
        else:
            logger.info(f"✅ Tracker test successful")
            self.use_tracking = True
        
        # Load model
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
        
        # Load embeddings and usernames
        self.embeddings, self.names = self.load_faceslist()
        if self.embeddings is None or self.names is None:
            raise Exception("Cannot load face data. Please register users first.")
        
        # Debug info
        logger.info(f"Embeddings shape: {self.embeddings.shape}")
        logger.info(f"Embeddings dtype: {self.embeddings.dtype}")
        logger.info(f"Names type: {type(self.names)}")
        logger.info(f"First few names: {self.names[:3] if len(self.names) > 0 else 'No names'}")
        
        # Tracker states cho từng client
        self.client_trackers = {}  # client_id -> tracker_states
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_inferences_run': 0,
            'successful_recognitions': 0,
            'quality_rejections': 0,
            'similarity_rejections': 0,
            'cached_results': 0
        }
        
        logger.info(f"✅ Server initialized with {len(self.names)} registered users")

    def load_faceslist(self):
        """Load existing embeddings and usernames"""
        import os
        
        DATA_PATH = './data'
        embeddings = None
        usernames = None
        
        # Load embeddings
        if self.device.type == 'cpu':
            embedding_file = os.path.join(DATA_PATH, "faceslistCPU.pth")
        else:
            embedding_file = os.path.join(DATA_PATH, "faceslist.pth")
        
        if os.path.exists(embedding_file):
            embeddings = torch.load(embedding_file, map_location=self.device)
            # Ensure embeddings are on correct device and have correct dtype
            embeddings = embeddings.to(self.device).float()
            logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
        else:
            logger.error(f"❌ Cannot find embeddings file: {embedding_file}")
        
        # Load usernames
        username_file = os.path.join(DATA_PATH, "usernames.npy")
        if os.path.exists(username_file):
            usernames = np.load(username_file, allow_pickle=True)
            # Convert to list if it's a numpy array
            if isinstance(usernames, np.ndarray):
                usernames = usernames.tolist()
            # Ensure all usernames are strings
            usernames = [str(name) for name in usernames]
            logger.info(f"✅ Loaded usernames: {usernames}")
        else:
            logger.error(f"❌ Cannot find usernames file: {username_file}")
        
        return embeddings, usernames

    def decode_frame_from_base64(self, base64_string):
        """Decode base64 string to OpenCV frame"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to numpy array with explicit dtype
            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
            
            # Decode to OpenCV image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode image from base64 - cv2.imdecode returned None")
                return None
            
            # Ensure frame is in correct format
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.error(f"Invalid frame shape: {frame.shape}")
                return None
                
            return frame
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None

    def encode_frame_to_base64(self, frame):
        """Encode OpenCV frame to base64 string"""
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{frame_base64}"
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None

    def validate_face_quality(self, face_tensor, frame_roi, mode='inference'):
        """Kiểm tra chất lượng face trước khi nhận diện"""
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

    def inference(self, face_tensor, threshold=None):
        """Face recognition inference"""
        if face_tensor is None:
            return -1, -1, 0.0
        
        if threshold is None:
            threshold = FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD
        
        similarity_threshold = 1 - threshold
        
        with torch.no_grad():
            detect_embeds = self.model(face_tensor.unsqueeze(0).to(self.device))
        
        # Calculate cosine similarity
        detect_embeds_norm = torch.nn.functional.normalize(detect_embeds, p=2, dim=1)
        local_embeds_norm = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)
        
        similarity = torch.mm(detect_embeds_norm, local_embeds_norm.t())
        max_sim, embed_idx = torch.max(similarity, dim=1)
        
        distance = 1 - max_sim
        confidence_percentage = max_sim.item() * 100
        
        if distance > threshold:
            return -1, -1, confidence_percentage
        else:
            return embed_idx, distance, confidence_percentage

    class TrackerState:
        def __init__(self, tracker_id, bbox=None, frame=None):
            self.tracker_id = tracker_id
            self.tracker = None
            self.bbox = bbox
            self.tracking_active = False
            
            # Try to create and initialize tracker if bbox provided
            if bbox is not None and frame is not None:
                self.tracker = create_tracker()
                if self.tracker is not None:
                    try:
                        success = self.tracker.init(frame, bbox)
                        if success:
                            self.tracking_active = True
                        else:
                            logger.debug(f"Failed to initialize tracker for ID {tracker_id}")
                    except Exception as e:
                        logger.debug(f"Tracker init failed: {e}")
                        self.tracker = None
            
            self.recognition_result = None
            self.last_inference_frame = 0
            self.stable_count = 0
            self.is_stable = False
            self.consecutive_failures = 0
            
        def update_tracker(self, frame):
            """Update tracker position, returns success and bbox"""
            if self.tracker is None or not self.tracking_active:
                return False, self.bbox
                
            try:
                success, bbox = self.tracker.update(frame)
                if success:
                    self.bbox = bbox
                    return True, bbox
                else:
                    self.tracking_active = False
                    return False, self.bbox
            except Exception as e:
                logger.debug(f"Tracker update failed: {e}")
                self.tracking_active = False
                return False, self.bbox
            
        def should_run_inference(self, current_frame, inference_interval=3):
            if self.recognition_result is None or not self.is_stable:
                return (current_frame - self.last_inference_frame) >= max(1, inference_interval // 2)
            return (current_frame - self.last_inference_frame) >= inference_interval
        
        def update_recognition_result(self, name, confidence, status):
            new_result = {"name": name, "confidence": confidence, "status": status}
            
            if (self.recognition_result is not None and 
                self.recognition_result["name"] == name and 
                abs(self.recognition_result["confidence"] - confidence) < 10.0):
                self.stable_count += 1
            else:
                self.stable_count = 0
                self.is_stable = False
            
            if self.stable_count >= 3:
                self.is_stable = True
                
            self.recognition_result = new_result
            self.consecutive_failures = 0
            
        def mark_inference_failure(self):
            self.consecutive_failures += 1
            if self.consecutive_failures >= 5:
                self.recognition_result = None
                self.is_stable = False
                self.stable_count = 0

    async def process_frame(self, client_id, frame, frame_count):
        """Process a single frame for face recognition"""
        if client_id not in self.client_trackers:
            self.client_trackers[client_id] = []
        
        tracker_states = self.client_trackers[client_id]
        detection_interval = FaceRecognitionConfig.DETECTION_INTERVAL
        recognition_interval = 3
        
        # Detection phase - run every N frames or when no active trackers
        need_detection = (len(tracker_states) == 0 or 
                         frame_count % detection_interval == 0 or
                         not any(ts.tracking_active for ts in tracker_states))
        
        if need_detection:
            try:
                boxes, _, points_list = self.mtcnn.detect(frame, landmarks=True)
                
                if boxes is not None and len(boxes) > 0:
                    new_tracker_states = []
                    
                    for i, box in enumerate(boxes):
                        bbox = (int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1]))
                        
                        # Create new tracker state
                        tracker_state = self.TrackerState(
                            tracker_id=len(new_tracker_states), 
                            bbox=bbox, 
                            frame=frame if self.use_tracking else None
                        )
                        new_tracker_states.append(tracker_state)
                    
                    tracker_states = new_tracker_states
                    self.client_trackers[client_id] = tracker_states
                    
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                # Continue with existing trackers
        
        # Update trackers và nhận diện
        active_trackers = []
        recognition_results = []
        
        for tracker_state in tracker_states:
            # Update tracker position
            if self.use_tracking and tracker_state.tracking_active:
                success, bbox = tracker_state.update_tracker(frame)
                if not success:
                    # Tracker failed, skip this tracker
                    continue
            else:
                # Use stored bbox if no tracking
                bbox = tracker_state.bbox
                if bbox is None:
                    continue
            
            active_trackers.append(tracker_state)
            x, y, w, h = [int(v) for v in bbox]
            
            # Ensure bbox is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            
            # Check if we should run inference
            should_infer = tracker_state.should_run_inference(frame_count, recognition_interval)
            
            if should_infer:
                try:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        
                        try:
                            face_tensor = self.mtcnn(face_rgb)
                        except (RuntimeError, Exception) as e:
                            logger.debug(f"MTCNN failed for face ROI: {e}")
                            face_tensor = None
                        
                        if face_tensor is not None:
                            self.stats['total_frames_processed'] += 1
                            self.stats['total_inferences_run'] += 1
                            tracker_state.last_inference_frame = frame_count
                            
                            is_good_quality, quality_score = self.validate_face_quality(face_tensor, face_rgb, mode='inference')
                            
                            if is_good_quality:
                                idx, distance, confidence_percentage = self.inference(face_tensor)
                                confidence_percentage = confidence_percentage / 100
                                
                                if idx != -1:
                                    tracker_state.update_recognition_result(
                                        name=self.names[idx], 
                                        confidence=confidence_percentage, 
                                        status="recognized"
                                    )
                                    self.stats['successful_recognitions'] += 1
                                else:
                                    tracker_state.update_recognition_result(
                                        name="Unknown", 
                                        confidence=confidence_percentage, 
                                        status="unknown"
                                    )
                                    self.stats['similarity_rejections'] += 1
                            else:
                                tracker_state.update_recognition_result(
                                    name="Poor Quality", 
                                    confidence=quality_score * 100, 
                                    status="poor_quality"
                                )
                                self.stats['quality_rejections'] += 1
                        else:
                            tracker_state.mark_inference_failure()
                except Exception as e:
                    logger.debug(f"Inference failed for tracker {tracker_state.tracker_id}: {e}")
                    tracker_state.mark_inference_failure()
            else:
                self.stats['cached_results'] += 1
            
            # Add recognition result với bounding box
            if tracker_state.recognition_result:
                result = tracker_state.recognition_result.copy()
                result['bbox'] = {'x': x, 'y': y, 'width': w, 'height': h}
                result['tracker_id'] = tracker_state.tracker_id
                result['is_stable'] = tracker_state.is_stable
                recognition_results.append(result)
            
            # Vẽ bounding box lên frame
            if tracker_state.recognition_result:
                result = tracker_state.recognition_result
                
                if result["status"] == "recognized":
                    color = FaceRecognitionConfig.COLOR_RECOGNIZED
                    text = f"{result['name']} ({result['confidence']:.1f}%)"
                    if tracker_state.is_stable:
                        text += " ✓"
                elif result["status"] == "unknown":
                    color = FaceRecognitionConfig.COLOR_UNKNOWN
                    text = f"Unknown ({result['confidence']:.1f}%)"
                else:  # poor_quality
                    color = FaceRecognitionConfig.COLOR_POOR_QUALITY
                    text = f"Poor Quality ({result['confidence']:.1f})"
                
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, FaceRecognitionConfig.BBOX_THICKNESS)
                frame = cv2.putText(frame, text, (x, y-10), 
                                  FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE, 
                                  color, FaceRecognitionConfig.TEXT_THICKNESS)
            else:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), FaceRecognitionConfig.BBOX_THICKNESS)
                frame = cv2.putText(frame, "Processing...", (x, y-10), 
                                  FaceRecognitionConfig.TEXT_FONT, FaceRecognitionConfig.TEXT_SCALE, 
                                  (255, 255, 255), FaceRecognitionConfig.TEXT_THICKNESS)
        
        self.client_trackers[client_id] = active_trackers
        
        return frame, recognition_results

    def identify_bbox(self, recognition_results, send_bbox):
        """Identify bbox from recognition results"""
        for result in recognition_results:
            result_bbox = {}
            # Reposition
            result_bbox['xmin'] = result['bbox']['x'] + send_bbox['xmin']
            result_bbox['ymin'] = result['bbox']['y'] + send_bbox['ymin']
            result_bbox['xmax'] = result_bbox['xmin'] + result['bbox']['width']
            result_bbox['ymax'] = result_bbox['ymin'] + result['bbox']['height']

            result_bbox['xmin'] = float(result_bbox['xmin'])
            result_bbox['ymin'] = float(result_bbox['ymin'])
            result_bbox['xmax'] = float(result_bbox['xmax'])
            result_bbox['ymax'] = float(result_bbox['ymax'])
            # Scale
            # result_bbox['xmin'] = max(0, min(1, result_bbox['xmin'] / send_bbox['width']))
            # result_bbox['ymin'] = max(0, min(1, result_bbox['ymin'] / send_bbox['height']))
            # result_bbox['xmax'] = max(0, min(1, result_bbox['xmax'] / send_bbox['width']))
            # result_bbox['ymax'] = max(0, min(1, result_bbox['ymax'] / send_bbox['height']))
            result['bbox'] = result_bbox
        return recognition_results

    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_id = id(websocket)  # Unique client ID
        logger.info(f"New client connected: {client_id}")
        
        # Initialize client tracker
        self.client_trackers[client_id] = []
        frame_count = 0
        
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Validate message structure
                    if not isinstance(data, dict) or 'type' not in data:
                        logger.warning(f"Invalid message structure from client {client_id}")
                        continue
                    
                    if data.get('type') == 'frame':
                        frame_count += 1
                        
                        # Decode frame from base64
                        frame_base64 = data.get('frame')
                        if not frame_base64 or not isinstance(frame_base64, str):
                            logger.warning(f"Invalid frame data from client {client_id}")
                            continue
                            
                        # Read tracker_id from client (optional)
                        incoming_tracker_id = data.get('tracker_id', None)
                        if incoming_tracker_id is None:
                            # Default to 1 if not provided as per client assumption
                            incoming_tracker_id = 0
                        
                        frame = self.decode_frame_from_base64(frame_base64)
                        if frame is None:
                            logger.warning(f"Failed to decode frame from client {client_id}")
                            continue
                        
                        # Validate frame dimensions
                        if frame.shape[0] < 50 and frame.shape[1] < 50:
                            logger.warning(f"Frame too small from client {client_id}: {frame.shape}")
                            continue
                        
                        # Process frame
                        start_time = time.time()
                        try:
                            processed_frame, recognition_results = await self.process_frame(client_id, frame, frame_count)
                            recognition_results = self.identify_bbox(recognition_results, data.get('bbox', {}))
                            processing_time = time.time() - start_time
                            
                            # Encode processed frame
                            processed_frame_base64 = self.encode_frame_to_base64(processed_frame)

                            # Send response
                            response = {
                                'type': 'result',
                                'frame_id': data.get('frame_id', frame_count),
                                'recognition_results': recognition_results,
                                'processing_time': processing_time,
                                'stats': self.stats.copy(),
                                'tracker_id': incoming_tracker_id
                            }
                            
                            await websocket.send(json.dumps(response))
                            
                            # Log debug information about response
                            logger.debug(f"Response sent to client {client_id}: frame_id={data.get('frame_id', frame_count)}, processing_time={processing_time}")
                        except Exception as e:
                            logger.error(f"Error processing frame for client {client_id}: {e}")
                            # Send error response
                            error_response = {
                                'type': 'error',
                                'error': f'Processing failed: {str(e)}',
                                'frame_id': data.get('frame_id', frame_count)
                            }
                            await websocket.send(json.dumps(error_response))
                            
                            # Log debug information about error response
                            logger.debug(f"Error response sent to client {client_id}: {error_response}")
                    elif data.get('type') == 'get_stats':
                        # Send current statistics
                        await websocket.send(json.dumps({
                            'type': 'stats',
                            'stats': self.stats.copy()
                        }))
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message from client {client_id}: {e}")
                    try:
                        error_response = {
                            'type': 'error',
                            'error': f'Message handling failed: {str(e)}'
                        }
                        await websocket.send(json.dumps(error_response))
                    except:
                        pass  # Connection might be closed
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
        finally:
            # Cleanup client data
            if client_id in self.client_trackers:
                del self.client_trackers[client_id]

    async def start_server(self, host='localhost', port=8764):
        """Start the WebSocket server"""
        logger.info(f"Starting Face Recognition WebSocket Server on {host}:{port}")
        
        # For newer versions of websockets library
        try:
            import websockets.server
            async with websockets.server.serve(self.handle_client, host, port):
                logger.info("✅ Server started successfully")
                await asyncio.Future()  # Run forever
        except ImportError:
            # Fallback for older versions
            async with websockets.serve(self.handle_client, host, port):
                logger.info("✅ Server started successfully") 
                await asyncio.Future()  # Run forever

# Main execution
async def main():
    try:
        server = FaceRecognitionWebSocketServer()
        await server.start_server(host='0.0.0.0', port=8764)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())