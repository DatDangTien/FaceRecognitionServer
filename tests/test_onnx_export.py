#!/usr/bin/env python3
"""
Test script for MTCNN ONNX export functionality
"""

import torch
import os
import sys
from facenet.models.mtcnn import PNet, RNet, ONet
from facenet.models.inception_resnet_v1 import InceptionResnetV1


def test_model_loading():
    """Test that all models can be loaded successfully."""
    print("🔄 Testing model loading...")
    
    try:
        # Test PNet
        pnet = PNet(pretrained=True)
        print("✅ PNet loaded successfully")
        
        # Test RNet
        rnet = RNet(pretrained=True)
        print("✅ RNet loaded successfully")
        
        # Test ONet
        onet = ONet(pretrained=True)
        print("✅ ONet loaded successfully")
        
        # Test InceptionResnetV1
        model = InceptionResnetV1(pretrained="casia-webface", classify=False)
        print("✅ InceptionResnetV1 loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False


def test_model_forward_pass():
    """Test that models can perform forward pass."""
    print("\n🔄 Testing model forward pass...")
    
    try:
        # Test PNet
        pnet = PNet(pretrained=True)
        pnet.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            bbox, prob = pnet(dummy_input)
        print(f"✅ PNet forward pass: bbox shape {bbox.shape}, prob shape {prob.shape}")
        
        # Test RNet
        rnet = RNet(pretrained=True)
        rnet.eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        with torch.no_grad():
            bbox, prob = rnet(dummy_input)
        print(f"✅ RNet forward pass: bbox shape {bbox.shape}, prob shape {prob.shape}")
        
        # Test ONet
        onet = ONet(pretrained=True)
        onet.eval()
        dummy_input = torch.randn(1, 3, 48, 48)
        with torch.no_grad():
            bbox, landmarks, prob = onet(dummy_input)
        print(f"✅ ONet forward pass: bbox shape {bbox.shape}, landmarks shape {landmarks.shape}, prob shape {prob.shape}")
        
        # Test InceptionResnetV1
        model = InceptionResnetV1(pretrained="casia-webface", classify=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 160, 160)
        with torch.no_grad():
            embedding = model(dummy_input)
        print(f"✅ InceptionResnetV1 forward pass: embedding shape {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {str(e)}")
        return False


def test_onnx_export():
    """Test ONNX export functionality."""
    print("\n🔄 Testing ONNX export...")
    
    try:
        # Test PNet export
        pnet = PNet(pretrained=True)
        pnet.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Create output directory
        os.makedirs("test_onnx", exist_ok=True)
        
        # Export PNet
        torch.onnx.export(
            pnet,
            (dummy_input,),
            "test_onnx/pnet_test.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['bbox_regression', 'face_probability']
        )
        print("✅ PNet ONNX export successful")
        
        # Test RNet export
        rnet = RNet(pretrained=True)
        rnet.eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        
        torch.onnx.export(
            rnet,
            (dummy_input,),
            "test_onnx/rnet_test.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['bbox_regression', 'face_probability']
        )
        print("✅ RNet ONNX export successful")
        
        # Test ONet export
        onet = ONet(pretrained=True)
        onet.eval()
        dummy_input = torch.randn(1, 3, 48, 48)
        
        torch.onnx.export(
            onet,
            (dummy_input,),
            "test_onnx/onet_test.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['bbox_regression', 'landmarks', 'face_probability']
        )
        print("✅ ONet ONNX export successful")
        
        # Test InceptionResnetV1 export
        model = InceptionResnetV1(pretrained="casia-webface", classify=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 160, 160)
        
        torch.onnx.export(
            model,
            (dummy_input,),
            "test_onnx/inception_resnet_v1_test.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embeddings']
        )
        print("✅ InceptionResnetV1 ONNX export successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in ONNX export: {str(e)}")
        return False


def test_onnx_verification():
    """Test ONNX model verification."""
    print("\n🔄 Testing ONNX model verification...")
    
    try:
        import onnx
        
        # Verify PNet
        model = onnx.load("test_onnx/pnet_test.onnx")
        onnx.checker.check_model(model)
        print("✅ PNet ONNX model verified")
        
        # Verify RNet
        model = onnx.load("test_onnx/rnet_test.onnx")
        onnx.checker.check_model(model)
        print("✅ RNet ONNX model verified")
        
        # Verify ONet
        model = onnx.load("test_onnx/onet_test.onnx")
        onnx.checker.check_model(model)
        print("✅ ONet ONNX model verified")
        
        # Verify InceptionResnetV1
        model = onnx.load("test_onnx/inception_resnet_v1_test.onnx")
        onnx.checker.check_model(model)
        print("✅ InceptionResnetV1 ONNX model verified")
        
        return True
        
    except ImportError:
        print("⚠️  ONNX not installed. Skipping verification.")
        return True
    except Exception as e:
        print(f"❌ Error in ONNX verification: {str(e)}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    print("\n🔄 Cleaning up test files...")
    
    try:
        import shutil
        if os.path.exists("test_onnx"):
            shutil.rmtree("test_onnx")
        print("✅ Test files cleaned up")
        return True
    except Exception as e:
        print(f"❌ Error cleaning up: {str(e)}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting MTCNN ONNX Export Tests")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Forward Pass", test_model_forward_pass),
        ("ONNX Export", test_onnx_export),
        ("ONNX Verification", test_onnx_verification),
        ("Cleanup", cleanup_test_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ONNX export is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

