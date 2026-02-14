#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ##################################################
    ## SynthPose Skeleton (52 Keypoints)            ##
    ##################################################
    
    SynthPose skeleton definition for 52 keypoints:
    - 0-16: COCO17 standard keypoints
    - 17-51: Additional 35 anatomical markers
    
    This module provides:
    - Keypoint name definitions
    - Skeleton tree structure for Sports2D visualization
    - Skeleton links for drawing
'''

import numpy as np
from anytree import Node


## AUTHORSHIP INFORMATION
__author__ = "Sports2D Contributors"
__copyright__ = "Copyright 2024, Sports2D"
__license__ = "BSD 3-Clause License"


## SynthPose 52 Keypoint Names
# Based on stanfordmimi/synthpose-vitpose-huge-hf model output
SYNTHPOSE_KEYPOINT_NAMES = [
    # COCO17 standard keypoints (0-16)
    'Nose',           # 0
    'LEye',           # 1
    'REye',           # 2
    'LEar',           # 3
    'REar',           # 4
    'LShoulder',      # 5
    'RShoulder',      # 6
    'LElbow',         # 7
    'RElbow',         # 8
    'LWrist',         # 9
    'RWrist',         # 10
    'LHip',           # 11
    'RHip',           # 12
    'LKnee',          # 13
    'RKnee',          # 14
    'LAnkle',         # 15
    'RAnkle',         # 16
    # Additional anatomical markers (17-51)
    'Sternum',        # 17 - Sternum/Neck area
    'RShoulder2',     # 18 - Right shoulder marker
    'LShoulder2',     # 19 - Left shoulder marker
    'RElbowLat',      # 20 - Right lateral elbow
    'LElbowLat',      # 21 - Left lateral elbow
    'RElbowMed',      # 22 - Right medial elbow
    'LElbowMed',      # 23 - Left medial elbow
    'RWristLat',      # 24 - Right lateral wrist
    'LWristLat',      # 25 - Left lateral wrist
    'RWristMed',      # 26 - Right medial wrist
    'LWristMed',      # 27 - Left medial wrist
    'RASIS',          # 28 - Right ASIS
    'LASIS',          # 29 - Left ASIS
    'RPSIS',          # 30 - Right PSIS
    'LPSIS',          # 31 - Left PSIS
    'RKneeLat',       # 32 - Right lateral knee
    'LKneeLat',       # 33 - Left lateral knee
    'RKneeMed',       # 34 - Right medial knee
    'LKneeMed',       # 35 - Left medial knee
    'RAnkleLat',      # 36 - Right lateral ankle
    'LAnkleLat',      # 37 - Left lateral ankle
    'RAnkleMed',      # 38 - Right medial ankle
    'LAnkleMed',      # 39 - Left medial ankle
    'R5Meta',         # 40 - Right 5th metatarsal (small toe area)
    'L5Meta',         # 41 - Left 5th metatarsal (small toe area)
    'RToe',           # 42 - Right toe tip
    'LToe',           # 43 - Left toe tip
    'RBigToe',        # 44 - Right big toe
    'LBigToe',        # 45 - Left big toe
    'LHeel',          # 46 - Left calcaneus (heel)
    'RHeel',          # 47 - Right calcaneus (heel)
    'C7',             # 48 - C7 vertebra
    'L2',             # 49 - L2 vertebra
    'T11',            # 50 - T11 vertebra
    'T6',             # 51 - T6 vertebra
]


def create_synthpose_skeleton():
    '''
    Create a skeleton tree structure for SynthPose 52 keypoints.
    Used by Sports2D for visualization and keypoint organization.
    
    OUTPUT:
    - Root node of the skeleton tree (anytree.Node)
    '''
    
    # Create Hip as root (midpoint between LHip and RHip - computed virtually)
    Hip = Node("Hip", id=None)  # Virtual node, will be computed as midpoint
    
    # Spine chain
    L2 = Node("L2", id=49, parent=Hip)
    T11 = Node("T11", id=50, parent=L2)
    T6 = Node("T6", id=51, parent=T11)
    C7 = Node("C7", id=48, parent=T6)
    Sternum = Node("Sternum", id=17, parent=C7)
    
    # Right leg
    RHip = Node("RHip", id=12, parent=Hip)
    RASIS = Node("RASIS", id=28, parent=RHip)
    RPSIS = Node("RPSIS", id=30, parent=RHip)
    RKnee = Node("RKnee", id=14, parent=RHip)
    RKneeLat = Node("RKneeLat", id=32, parent=RKnee)
    RKneeMed = Node("RKneeMed", id=34, parent=RKnee)
    RAnkle = Node("RAnkle", id=16, parent=RKnee)
    RAnkleLat = Node("RAnkleLat", id=36, parent=RAnkle)
    RAnkleMed = Node("RAnkleMed", id=38, parent=RAnkle)
    RBigToe = Node("RBigToe", id=44, parent=RAnkle)
    RToe = Node("RToe", id=42, parent=RBigToe)
    R5Meta = Node("R5Meta", id=40, parent=RAnkle)
    RHeel = Node("RHeel", id=47, parent=RAnkle)
    
    # Left leg
    LHip = Node("LHip", id=11, parent=Hip)
    LASIS = Node("LASIS", id=29, parent=LHip)
    LPSIS = Node("LPSIS", id=31, parent=LHip)
    LKnee = Node("LKnee", id=13, parent=LHip)
    LKneeLat = Node("LKneeLat", id=33, parent=LKnee)
    LKneeMed = Node("LKneeMed", id=35, parent=LKnee)
    LAnkle = Node("LAnkle", id=15, parent=LKnee)
    LAnkleLat = Node("LAnkleLat", id=37, parent=LAnkle)
    LAnkleMed = Node("LAnkleMed", id=39, parent=LAnkle)
    LBigToe = Node("LBigToe", id=45, parent=LAnkle)
    LToe = Node("LToe", id=43, parent=LBigToe)
    L5Meta = Node("L5Meta", id=41, parent=LAnkle)
    LHeel = Node("LHeel", id=46, parent=LAnkle)
    
    # Head
    Nose = Node("Nose", id=0, parent=Sternum)
    LEye = Node("LEye", id=1, parent=Nose)
    REye = Node("REye", id=2, parent=Nose)
    LEar = Node("LEar", id=3, parent=LEye)
    REar = Node("REar", id=4, parent=REye)
    
    # Right arm
    RShoulder = Node("RShoulder", id=6, parent=Sternum)
    RShoulder2 = Node("RShoulder2", id=18, parent=RShoulder)
    RElbow = Node("RElbow", id=8, parent=RShoulder)
    RElbowLat = Node("RElbowLat", id=20, parent=RElbow)
    RElbowMed = Node("RElbowMed", id=22, parent=RElbow)
    RWrist = Node("RWrist", id=10, parent=RElbow)
    RWristLat = Node("RWristLat", id=24, parent=RWrist)
    RWristMed = Node("RWristMed", id=26, parent=RWrist)
    
    # Left arm
    LShoulder = Node("LShoulder", id=5, parent=Sternum)
    LShoulder2 = Node("LShoulder2", id=19, parent=LShoulder)
    LElbow = Node("LElbow", id=7, parent=LShoulder)
    LElbowLat = Node("LElbowLat", id=21, parent=LElbow)
    LElbowMed = Node("LElbowMed", id=23, parent=LElbow)
    LWrist = Node("LWrist", id=9, parent=LElbow)
    LWristLat = Node("LWristLat", id=25, parent=LWrist)
    LWristMed = Node("LWristMed", id=27, parent=LWrist)
    
    return Hip


# Skeleton connection lines for visualization (SynthPose 52 format)
# Format: (start_keypoint_id, end_keypoint_id)
SYNTHPOSE_SKELETON_LINKS = [
    # Face
    (0, 1), (0, 2),      # Nose to Eyes
    (1, 3), (2, 4),      # Eyes to Ears
    
    # Spine
    (17, 48),            # Sternum to C7
    (48, 51),            # C7 to T6
    (51, 50),            # T6 to T11
    (50, 49),            # T11 to L2
    
    # Shoulders
    (17, 5), (17, 6),    # Sternum to Shoulders
    (5, 19), (6, 18),    # Shoulders to Shoulder2 markers
    
    # Right arm
    (6, 8),              # RShoulder to RElbow
    (8, 20), (8, 22),    # RElbow to lateral/medial
    (8, 10),             # RElbow to RWrist
    (10, 24), (10, 26),  # RWrist to lateral/medial
    
    # Left arm
    (5, 7),              # LShoulder to LElbow
    (7, 21), (7, 23),    # LElbow to lateral/medial
    (7, 9),              # LElbow to LWrist
    (9, 25), (9, 27),    # LWrist to lateral/medial
    
    # Hip connections
    (49, 11), (49, 12),  # L2 to Hips
    (11, 12),            # LHip to RHip
    (11, 29), (12, 28),  # Hips to ASIS
    (11, 31), (12, 30),  # Hips to PSIS
    
    # Right leg
    (12, 14),            # RHip to RKnee
    (14, 32), (14, 34),  # RKnee to lateral/medial
    (14, 16),            # RKnee to RAnkle
    (16, 36), (16, 38),  # RAnkle to lateral/medial
    (16, 44),            # RAnkle to RBigToe
    (44, 42),            # RBigToe to RToe
    (16, 40),            # RAnkle to R5Meta
    (16, 47),            # RAnkle to RHeel
    
    # Left leg
    (11, 13),            # LHip to LKnee
    (13, 33), (13, 35),  # LKnee to lateral/medial
    (13, 15),            # LKnee to LAnkle
    (15, 37), (15, 39),  # LAnkle to lateral/medial
    (15, 45),            # LAnkle to LBigToe
    (45, 43),            # LBigToe to LToe
    (15, 41),            # LAnkle to L5Meta
    (15, 46),            # LAnkle to LHeel
]


# ============================================================
# HALPE26 MAPPING - SINGLE SOURCE OF TRUTH
# ============================================================
# Halpe26 bodywithfeet = COCO17 body (0-16) + foot keypoints (40-47)
# These keypoints will be visualized as COLORED CIRCLES
# All other anatomical markers (17-39, 48-51) will be 1/2 size DIAMONDS

# Indices of SynthPose keypoints that correspond to HALPE26 bodywithfeet
SYNTHPOSE_HALPE26_BODYWITHFEET_INDICES = {
    # COCO17 body keypoints (0-16)
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    # Foot keypoints (40-47)
    40,  # R5Meta (right 5th metatarsal / small toe area)
    41,  # L5Meta (left 5th metatarsal / small toe area)
    42,  # RToe (right toe tip)
    43,  # LToe (left toe tip)
    44,  # RBigToe (right big toe)
    45,  # LBigToe (left big toe)
    46,  # LHeel (left calcaneus/heel)
    47,  # RHeel (right calcaneus/heel)
}

# Names of HALPE26 bodywithfeet keypoints for name-based lookup
SYNTHPOSE_HALPE26_BODYWITHFEET_NAMES = {
    'Nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle',
    'R5Meta', 'L5Meta', 'RToe', 'LToe', 'RBigToe', 'LBigToe', 'LHeel', 'RHeel'
}


# Color palette for 52 keypoints
# COCO17 (0-16): standard colors, Anatomical markers (17-51): lighter variants
SYNTHPOSE_KEYPOINT_COLORS = [
    # COCO17 (0-16)
    (255, 0, 0),     # 0 Nose - Red
    (255, 85, 0),    # 1 LEye
    (255, 170, 0),   # 2 REye
    (255, 255, 0),   # 3 LEar
    (170, 255, 0),   # 4 REar
    (85, 255, 0),    # 5 LShoulder
    (0, 255, 0),     # 6 RShoulder
    (0, 255, 85),    # 7 LElbow
    (0, 255, 170),   # 8 RElbow
    (0, 255, 255),   # 9 LWrist
    (0, 170, 255),   # 10 RWrist
    (0, 85, 255),    # 11 LHip
    (0, 0, 255),     # 12 RHip
    (85, 0, 255),    # 13 LKnee
    (170, 0, 255),   # 14 RKnee
    (255, 0, 255),   # 15 LAnkle
    (255, 0, 170),   # 16 RAnkle
    # Anatomical markers (17-51) - white/gray tones for distinction
    (200, 200, 200), # 17 Sternum
    (180, 255, 180), # 18 RShoulder2
    (180, 255, 180), # 19 LShoulder2
    (180, 255, 220), # 20 RElbowLat
    (180, 255, 220), # 21 LElbowLat
    (180, 255, 220), # 22 RElbowMed
    (180, 255, 220), # 23 LElbowMed
    (180, 255, 255), # 24 RWristLat
    (180, 255, 255), # 25 LWristLat
    (180, 255, 255), # 26 RWristMed
    (180, 255, 255), # 27 LWristMed
    (180, 180, 255), # 28 RASIS
    (180, 180, 255), # 29 LASIS
    (180, 180, 255), # 30 RPSIS
    (180, 180, 255), # 31 LPSIS
    (220, 180, 255), # 32 RKneeLat
    (220, 180, 255), # 33 LKneeLat
    (220, 180, 255), # 34 RKneeMed
    (220, 180, 255), # 35 LKneeMed
    (255, 180, 255), # 36 RAnkleLat
    (255, 180, 255), # 37 LAnkleLat
    (255, 180, 255), # 38 RAnkleMed
    (255, 180, 255), # 39 LAnkleMed
    (255, 180, 220), # 40 R5Meta
    (255, 180, 220), # 41 L5Meta
    (255, 200, 200), # 42 RToe
    (255, 200, 200), # 43 LToe
    (255, 220, 180), # 44 RBigToe
    (255, 220, 180), # 45 LBigToe
    (255, 255, 180), # 46 LHeel
    (255, 255, 180), # 47 RHeel
    (200, 200, 200), # 48 C7
    (200, 200, 200), # 49 L2
    (200, 200, 200), # 50 T11
    (200, 200, 200), # 51 T6
]


# ============================================================
# For backward compatibility - HALPE26 mapping (optional use)
# ============================================================

HALPE26_KEYPOINT_NAMES = [
    'Nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle',
    'Head', 'Neck', 'Hip', 'LBigToe', 'RBigToe', 'LSmallToe', 'RSmallToe', 'LHeel', 'RHeel'
]

# Mapping from SynthPose to HALPE_26 (for optional conversion)
SYNTHPOSE_TO_HALPE26_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
    17: 'head', 18: 17, 19: 'hip',
    20: 45, 21: 44, 22: 41, 23: 40, 24: 46, 25: 47
}


def map_synthpose_to_halpe26(keypoints_52, scores_52):
    '''
    Optional: Map SynthPose 52-keypoint output to HALPE_26 format.
    
    INPUTS:
    - keypoints_52: np.array shape (N_persons, 52, 2)
    - scores_52: np.array shape (N_persons, 52)
    
    OUTPUTS:
    - keypoints_26: np.array shape (N_persons, 26, 2)
    - scores_26: np.array shape (N_persons, 26)
    '''
    if keypoints_52.size == 0:
        return np.array([]).reshape(0, 26, 2), np.array([]).reshape(0, 26)
    
    n_persons = keypoints_52.shape[0]
    keypoints_26 = np.full((n_persons, 26, 2), np.nan)
    scores_26 = np.full((n_persons, 26), np.nan)
    
    # Direct mappings (COCO17)
    for i in range(17):
        keypoints_26[:, i, :] = keypoints_52[:, i, :]
        scores_26[:, i] = scores_52[:, i]
    
    # Head (17) - midpoint of eyes, offset up
    keypoints_26[:, 17, :] = (keypoints_52[:, 1, :] + keypoints_52[:, 2, :]) / 2
    eye_dist = np.linalg.norm(keypoints_52[:, 1, :] - keypoints_52[:, 2, :], axis=1, keepdims=True)
    keypoints_26[:, 17, 1:2] -= eye_dist * 0.8
    scores_26[:, 17] = (scores_52[:, 1] + scores_52[:, 2]) / 2
    
    # Neck (18) - use Sternum
    keypoints_26[:, 18, :] = keypoints_52[:, 17, :]
    scores_26[:, 18] = scores_52[:, 17]
    
    # Hip (19) - midpoint
    keypoints_26[:, 19, :] = (keypoints_52[:, 11, :] + keypoints_52[:, 12, :]) / 2
    scores_26[:, 19] = (scores_52[:, 11] + scores_52[:, 12]) / 2
    
    # Toes and heels
    keypoints_26[:, 20, :] = keypoints_52[:, 45, :]  # LBigToe
    scores_26[:, 20] = scores_52[:, 45]
    keypoints_26[:, 21, :] = keypoints_52[:, 44, :]  # RBigToe
    scores_26[:, 21] = scores_52[:, 44]
    keypoints_26[:, 22, :] = keypoints_52[:, 41, :]  # LSmallToe (L5Meta)
    scores_26[:, 22] = scores_52[:, 41]
    keypoints_26[:, 23, :] = keypoints_52[:, 40, :]  # RSmallToe (R5Meta)
    scores_26[:, 23] = scores_52[:, 40]
    keypoints_26[:, 24, :] = keypoints_52[:, 46, :]  # LHeel
    scores_26[:, 24] = scores_52[:, 46]
    keypoints_26[:, 25, :] = keypoints_52[:, 47, :]  # RHeel
    scores_26[:, 25] = scores_52[:, 47]
    
    return keypoints_26, scores_26


def create_synthpose_halpe26_skeleton():
    '''
    Create HALPE_26-compatible skeleton for backward compatibility.
    '''
    Hip = Node("Hip", id=19)
    
    RHip = Node("RHip", id=12, parent=Hip)
    RKnee = Node("RKnee", id=14, parent=RHip)
    RAnkle = Node("RAnkle", id=16, parent=RKnee)
    RBigToe = Node("RBigToe", id=21, parent=RAnkle)
    RSmallToe = Node("RSmallToe", id=23, parent=RBigToe)
    RHeel = Node("RHeel", id=25, parent=RAnkle)
    
    LHip = Node("LHip", id=11, parent=Hip)
    LKnee = Node("LKnee", id=13, parent=LHip)
    LAnkle = Node("LAnkle", id=15, parent=LKnee)
    LBigToe = Node("LBigToe", id=20, parent=LAnkle)
    LSmallToe = Node("LSmallToe", id=22, parent=LBigToe)
    LHeel = Node("LHeel", id=24, parent=LAnkle)
    
    Neck = Node("Neck", id=18, parent=Hip)
    Head = Node("Head", id=17, parent=Neck)
    Nose = Node("Nose", id=0, parent=Head)
    
    RShoulder = Node("RShoulder", id=6, parent=Neck)
    RElbow = Node("RElbow", id=8, parent=RShoulder)
    RWrist = Node("RWrist", id=10, parent=RElbow)
    
    LShoulder = Node("LShoulder", id=5, parent=Neck)
    LElbow = Node("LElbow", id=7, parent=LShoulder)
    LWrist = Node("LWrist", id=9, parent=LElbow)
    
    return Hip
