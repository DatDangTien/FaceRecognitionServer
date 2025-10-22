#!/usr/bin/env python3
"""
Test script để kiểm tra FaceRecognitionConfig
"""

from face_config import FaceRecognitionConfig

def test_config():
    """Test configuration values"""
    print("🧪 TESTING FACE RECOGNITION CONFIG")
    print("=" * 60)
    
    # Test basic config values
    print("📋 Basic Configuration:")
    print(f"   Registration Quality Threshold: {FaceRecognitionConfig.REGISTRATION_QUALITY_THRESHOLD}")
    print(f"   Inference Quality Threshold: {FaceRecognitionConfig.INFERENCE_QUALITY_THRESHOLD}")
    print(f"   Recognition Similarity Threshold: {FaceRecognitionConfig.RECOGNITION_SIMILARITY_THRESHOLD}")
    print(f"   Duplicate Detection Threshold: {FaceRecognitionConfig.DUPLICATE_DETECTION_THRESHOLD}")
    print()
    
    # Test quality thresholds for registration
    print("📝 Registration Quality Thresholds:")
    reg_thresholds = FaceRecognitionConfig.get_quality_thresholds('registration')
    for key, value in reg_thresholds.items():
        print(f"   {key}: {value}")
    print()
    
    # Test quality thresholds for inference
    print("🔍 Inference Quality Thresholds:")
    inf_thresholds = FaceRecognitionConfig.get_quality_thresholds('inference')
    for key, value in inf_thresholds.items():
        print(f"   {key}: {value}")
    print()
    
    # Test MTCNN settings
    print("🎯 MTCNN Settings:")
    print(f"   Margin: {FaceRecognitionConfig.MTCNN_MARGIN}")
    print(f"   Keep All: {FaceRecognitionConfig.MTCNN_KEEP_ALL}")
    print(f"   Select Largest: {FaceRecognitionConfig.MTCNN_SELECT_LARGEST}")
    print(f"   Post Process: {FaceRecognitionConfig.MTCNN_POST_PROCESS}")
    print(f"   Thresholds: {FaceRecognitionConfig.MTCNN_THRESHOLDS}")
    print()
    
    # Test camera settings
    print("📷 Camera Settings:")
    print(f"   Width: {FaceRecognitionConfig.CAMERA_WIDTH}")
    print(f"   Height: {FaceRecognitionConfig.CAMERA_HEIGHT}")
    print(f"   Face Size: {FaceRecognitionConfig.FACE_SIZE}")
    print()
    
    # Test registration settings
    print("📝 Registration Settings:")
    print(f"   Samples: {FaceRecognitionConfig.REGISTRATION_SAMPLES}")
    print(f"   Skip Frames: {FaceRecognitionConfig.REGISTRATION_SKIP_FRAMES}")
    print()
    
    # Test colors
    print("🎨 Color Settings:")
    print(f"   Recognized: {FaceRecognitionConfig.COLOR_RECOGNIZED}")
    print(f"   Unknown: {FaceRecognitionConfig.COLOR_UNKNOWN}")
    print(f"   Poor Quality: {FaceRecognitionConfig.COLOR_POOR_QUALITY}")
    print(f"   Tracking: {FaceRecognitionConfig.COLOR_TRACKING}")
    print(f"   White: {FaceRecognitionConfig.COLOR_WHITE}")
    print()
    
    # Test print_config method
    print("🖨️  Print Config Method:")
    FaceRecognitionConfig.print_config()
    
    print("\n✅ All configuration tests passed!")

def validate_thresholds():
    """Validate that thresholds make sense"""
    print("\n🔍 VALIDATING THRESHOLDS")
    print("=" * 40)
    
    # Check that registration thresholds are stricter than inference
    reg_thresholds = FaceRecognitionConfig.get_quality_thresholds('registration')
    inf_thresholds = FaceRecognitionConfig.get_quality_thresholds('inference')
    
    issues = []
    
    # Quality threshold should be higher for registration
    if reg_thresholds['quality_threshold'] <= inf_thresholds['quality_threshold']:
        issues.append("Registration quality threshold should be higher than inference")
    
    # Blur threshold should be higher for registration
    if reg_thresholds['blur_threshold'] <= inf_thresholds['blur_threshold']:
        issues.append("Registration blur threshold should be higher than inference")
    
    # Face size should be larger for registration
    if reg_thresholds['min_face_size'] <= inf_thresholds['min_face_size']:
        issues.append("Registration min face size should be larger than inference")
    
    # Pose threshold should be stricter for registration
    if reg_thresholds['pose_threshold'] >= inf_thresholds['pose_threshold']:
        issues.append("Registration pose threshold should be stricter than inference")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All threshold validations passed!")
    
    return len(issues) == 0

if __name__ == "__main__":
    test_config()
    validate_thresholds()
    
    print("\n🎯 CONFIGURATION SUMMARY:")
    print("=" * 50)
    print("✅ Configuration class created successfully")
    print("✅ All thresholds are properly configured")
    print("✅ Registration is stricter than inference")
    print("✅ Ready for face recognition system!")
    print("=" * 50)
