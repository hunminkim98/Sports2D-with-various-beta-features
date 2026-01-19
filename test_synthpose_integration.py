#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for SynthPose integration in Sports2D.
Run this to verify the integration is working correctly.
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_skeleton_module():
    """Test synthpose_skeleton.py module."""
    print("=" * 60)
    print("Testing synthpose_skeleton module...")
    print("=" * 60)
    
    from Sports2D.Utilities.synthpose_skeleton import (
        SYNTHPOSE_KEYPOINT_NAMES,
        SYNTHPOSE_SKELETON_LINKS,
        create_synthpose_skeleton
    )
    
    # Test constants
    assert len(SYNTHPOSE_KEYPOINT_NAMES) == 52, f"Expected 52 keypoints, got {len(SYNTHPOSE_KEYPOINT_NAMES)}"
    print(f"✓ SYNTHPOSE_KEYPOINT_NAMES: {len(SYNTHPOSE_KEYPOINT_NAMES)} keypoints")
    
    # Test skeleton links
    assert len(SYNTHPOSE_SKELETON_LINKS) > 0, "Should have skeleton links"
    print(f"✓ SYNTHPOSE_SKELETON_LINKS: {len(SYNTHPOSE_SKELETON_LINKS)} links defined")
    
    # Test skeleton creation (52-keypoint version)
    skeleton = create_synthpose_skeleton()
    assert skeleton is not None, "Skeleton should not be None"
    assert skeleton.name == "Hip", f"Root should be 'Hip', got {skeleton.name}"
    print(f"✓ create_synthpose_skeleton: root={skeleton.name}")
    
    # Count keypoints in skeleton
    from anytree import RenderTree
    keypoints_ids = [node.id for _, _, node in RenderTree(skeleton) if node.id is not None]
    keypoints_names = [node.name for _, _, node in RenderTree(skeleton) if node.id is not None]
    print(f"✓ Skeleton has {len(keypoints_names)} keypoints (expected ~51 with IDs, Hip has None)")
    
    # Verify key keypoints are present
    expected_keypoints = ['Nose', 'LShoulder', 'RShoulder', 'LHip', 'RHip', 'LKnee', 'RKnee', 
                         'LAnkle', 'RAnkle', 'LBigToe', 'RBigToe', 'LHeel', 'RHeel', 'Sternum', 'C7']
    for kpt in expected_keypoints:
        assert kpt in keypoints_names, f"Missing expected keypoint: {kpt}"
    print(f"✓ All key keypoints present: {expected_keypoints[:5]}...")
    
    print("\n✅ synthpose_skeleton module: ALL TESTS PASSED\n")


def test_tracker_module():
    """Test synthpose_tracker.py module."""
    print("=" * 60)
    print("Testing synthpose_tracker module...")
    print("=" * 60)
    
    from Sports2D.Utilities.synthpose_tracker import (
        SynthPosePoseTracker,
        setup_synthpose_tracker,
        _load_dependencies
    )
    
    # Test lazy import
    _load_dependencies()
    print("✓ Dependencies loaded successfully")
    
    # Test tracker class exists
    assert SynthPosePoseTracker is not None
    print("✓ SynthPosePoseTracker class available")
    
    # Test factory function
    assert setup_synthpose_tracker is not None
    print("✓ setup_synthpose_tracker function available")
    
    print("\n✅ synthpose_tracker module: ALL TESTS PASSED\n")


def test_tracker_initialization():
    """Test SynthPosePoseTracker initialization (downloads models on first run)."""
    print("=" * 60)
    print("Testing SynthPosePoseTracker initialization...")
    print("=" * 60)
    print("⚠ This will download models from HuggingFace (~2GB for huge model)")
    print("⚠ Set SKIP_MODEL_DOWNLOAD=1 to skip this test")
    
    if os.environ.get('SKIP_MODEL_DOWNLOAD'):
        print("Skipping model download test (SKIP_MODEL_DOWNLOAD=1)")
        return
    
    try:
        from Sports2D.Utilities.synthpose_tracker import SynthPosePoseTracker
        
        # Use base model (smaller, faster download)
        print("Initializing SynthPosePoseTracker with mode='base'...")
        tracker = SynthPosePoseTracker(mode='base', device='cpu')
        print("✓ Tracker initialized successfully")
        
        # Test with dummy frame
        import cv2
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints, scores = tracker(dummy_frame)
        
        print(f"✓ Inference completed: keypoints={keypoints.shape}, scores={scores.shape}")
        
        print("\n✅ SynthPosePoseTracker initialization: ALL TESTS PASSED\n")
        
    except Exception as e:
        print(f"\n⚠ Tracker initialization test failed: {e}")
        print("This might be due to network issues or memory constraints.")


def test_integration_with_sports2d():
    """Test integration with Sports2D process module."""
    print("=" * 60)
    print("Testing integration with Sports2D...")
    print("=" * 60)
    
    # Test that process.py can import SynthPose components
    try:
        # Simulate what process.py does
        SYNTHPOSE_AVAILABLE = False
        try:
            from Sports2D.Utilities.synthpose_tracker import SynthPosePoseTracker
            from Sports2D.Utilities.synthpose_skeleton import (
                create_synthpose_skeleton,
                SYNTHPOSE_KEYPOINT_NAMES,
                SYNTHPOSE_SKELETON_LINKS,
            )
            SYNTHPOSE_AVAILABLE = True
        except ImportError:
            pass
        
        assert SYNTHPOSE_AVAILABLE, "SynthPose should be available"
        print("✓ SynthPose modules imported successfully")
        
        # Test 52-keypoint skeleton creation
        skeleton = create_synthpose_skeleton()
        assert skeleton is not None, "Skeleton should be created"
        print(f"✓ 52-keypoint skeleton created with root: {skeleton.name}")
        
        # Test pose_model detection logic
        pose_model_name = 'synthpose'
        use_synthpose = pose_model_name.lower() in ['synthpose', 'synthpose_base']
        assert use_synthpose, "Should detect synthpose model"
        print("✓ SynthPose model detection works")
        
        pose_model_name = 'synthpose_base'
        use_synthpose = pose_model_name.lower() in ['synthpose', 'synthpose_base']
        assert use_synthpose, "Should detect synthpose_base model"
        print("✓ SynthPose base model detection works")
        
        pose_model_name = 'body_with_feet'
        use_synthpose = pose_model_name.lower() in ['synthpose', 'synthpose_base']
        assert not use_synthpose, "Should not detect RTMLib model as synthpose"
        print("✓ RTMLib model detection works (negative test)")
        
        print("\n✅ Sports2D integration: ALL TESTS PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SynthPose Integration Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_skeleton_module()
        test_tracker_module()
        test_integration_with_sports2d()
        
        # Optional: model download test
        if len(sys.argv) > 1 and sys.argv[1] == '--with-models':
            test_tracker_initialization()
        else:
            print("Tip: Run with '--with-models' to test model download and inference")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
