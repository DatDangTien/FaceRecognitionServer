#!/usr/bin/env python3
"""
Face Recognition WebSocket Client - CLI Version
Captures from 2 USB cameras and sends frames with tracker structure
"""

import cv2
import websocket
import json
import base64
import time
import threading
import argparse
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketClient:
    def __init__(self, server_url="ws://localhost:8764"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.cameras = {}
        self.running = False
        self.frame_id = 0
        self.stats = {
            'frames_sent': 0,
            'errors': 0,
            'start_time': None
        }

    def connect(self):
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to {self.server_url}...")
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                raise Exception("Failed to connect within timeout")
                
            logger.info("Connected to WebSocket server")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def on_open(self, ws):
        """WebSocket connection opened"""
        self.connected = True
        logger.info("WebSocket connection established")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            if data.get('type') == 'result':
                logger.info(f"Server response: {data.get('stats', {})}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.stats['errors'] += 1

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.connected = False
        logger.info("WebSocket connection closed")

    def initialize_cameras(self, camera_indices=[0, 1]):
        """Initialize USB cameras"""
        logger.info("Initializing cameras...")
        
        for idx in camera_indices:
            try:
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    logger.warning(f"Could not open camera {idx}")
                    continue
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test capture
                ret, frame = cap.read()
                if ret:
                    self.cameras[idx] = cap
                    logger.info(f"Camera {idx} initialized successfully")
                else:
                    logger.warning(f"Camera {idx} failed test capture")
                    cap.release()
                    
            except Exception as e:
                logger.error(f"Error initializing camera {idx}: {e}")
        
        if not self.cameras:
            logger.error("No cameras available!")
            return False
        
        logger.info(f"Successfully initialized {len(self.cameras)} cameras")
        return True

    def frame_to_base64(self, frame, quality=70):
        """Convert OpenCV frame to base64 JPEG"""
        try:
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_base64}"
            
        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return None

    def create_test_message(self):
        """Create test message with tracker structure"""
        bbox_images = []
        
        # Capture frames from available cameras
        camera_indices = list(self.cameras.keys())
        frames_captured = []
        
        for cam_idx in camera_indices:
            try:
                ret, frame = self.cameras[cam_idx].read()
                if ret:
                    frames_captured.append((cam_idx, frame))
                else:
                    logger.warning(f"Failed to capture from camera {cam_idx}")
            except Exception as e:
                logger.error(f"Error capturing from camera {cam_idx}: {e}")
        
        if not frames_captured:
            logger.error("No frames captured from cameras")
            return None
        
        # Create trackers based on available frames
        # For test case: assuming 2 trackers with potentially multiple frames per tracker
        tracker_assignments = {
            1: [],  # Tracker 1
            2: []   # Tracker 2
        }
        
        # Assign frames to trackers (for test case)
        for i, (cam_idx, frame) in enumerate(frames_captured):
            tracker_id = (i % 2) + 1  # Alternate between tracker 1 and 2
            
            # Convert frame to base64
            frame_base64 = self.frame_to_base64(frame)
            if frame_base64:
                tracker_item = {
                    'tracker_id': tracker_id,
                    'frame': frame_base64
                }
                tracker_assignments[tracker_id].append(tracker_item)
        
        # Flatten tracker assignments into bbox_images array
        for tracker_id, items in tracker_assignments.items():
            bbox_images.extend(items)
        
        # Add additional test frame for tracker 1 (to have 3 items total as requested)
        if len(bbox_images) == 2 and frames_captured:
            # Duplicate first frame with tracker_id 1
            first_frame = frames_captured[0][1]
            frame_base64 = self.frame_to_base64(first_frame)
            if frame_base64:
                additional_item = {
                    'tracker_id': 1,
                    'frame': frame_base64
                }
                bbox_images.append(additional_item)
        
        # Count unique trackers
        unique_trackers = set(item['tracker_id'] for item in bbox_images)
        count_tracker = len(unique_trackers)
        
        # Create message structure
        self.frame_id += 1
        message = {
            "type": "frame",
            "frame_id": self.frame_id,
            "timestamp": int(time.time() * 1000),  # Current timestamp in milliseconds
            "trackers": bbox_images,
            "count_tracker": count_tracker
        }
        
        return message

    def send_message(self, message):
        """Send message to WebSocket server"""
        if not self.connected or not self.ws:
            logger.error("Not connected to server")
            return False
        
        try:
            self.ws.send(json.dumps(message))
            self.stats['frames_sent'] += 1
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.stats['errors'] += 1
            return False

    def run_streaming(self, fps=10, duration=None):
        """Run streaming loop"""
        if not self.cameras:
            logger.error("No cameras initialized")
            return
        
        if not self.connected:
            logger.error("Not connected to server")
            return
        
        logger.info(f"Starting streaming at {fps} FPS...")
        if duration:
            logger.info(f"Will run for {duration} seconds")
        
        self.running = True
        self.stats['start_time'] = time.time()
        frame_interval = 1.0 / fps
        
        try:
            while self.running:
                start_time = time.time()
                
                # Create and send message
                message = self.create_test_message()
                if message:
                    success = self.send_message(message)
                    if success:
                        logger.info(f"Sent frame {message['frame_id']} with {len(message['trackers'])} tracker items, {message['count_tracker']} unique trackers")
                    else:
                        logger.error(f"Failed to send frame {message['frame_id']}")
                else:
                    logger.error("Failed to create message")
                
                # Check duration
                if duration and (time.time() - self.stats['start_time']) >= duration:
                    logger.info("Duration limit reached, stopping...")
                    break
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            self.running = False
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        # Release cameras
        for cam_idx, cap in self.cameras.items():
            try:
                cap.release()
                logger.info(f"Released camera {cam_idx}")
            except Exception as e:
                logger.error(f"Error releasing camera {cam_idx}: {e}")
        
        self.cameras.clear()
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
        
        # Print statistics
        self.print_stats()

    def print_stats(self):
        """Print streaming statistics"""
        if self.stats['start_time']:
            duration = time.time() - self.stats['start_time']
            avg_fps = self.stats['frames_sent'] / duration if duration > 0 else 0
            
            logger.info("=== Streaming Statistics ===")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Frames sent: {self.stats['frames_sent']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info(f"Average FPS: {avg_fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition WebSocket Client CLI")
    parser.add_argument("--server", "-s", default="ws://localhost:8764", 
                       help="WebSocket server URL (default: ws://localhost:8764)")
    parser.add_argument("--fps", "-f", type=int, default=10,
                       help="Frames per second (default: 10)")
    parser.add_argument("--duration", "-d", type=int,
                       help="Duration in seconds (default: unlimited)")
    parser.add_argument("--cameras", "-c", nargs="+", type=int, default=[0, 1],
                       help="Camera indices to use (default: 0 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create client
    client = WebSocketClient(args.server)
    
    try:
        # Connect to server
        if not client.connect():
            logger.error("Failed to connect to server")
            sys.exit(1)
        
        # Initialize cameras
        if not client.initialize_cameras(args.cameras):
            logger.error("Failed to initialize cameras")
            sys.exit(1)
        
        # Start streaming
        client.run_streaming(fps=args.fps, duration=args.duration)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main()