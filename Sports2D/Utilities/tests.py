'''
    ########################################
    ## Sports2D tests                     ##
    ########################################

    Check whether Sports2D still works after each code modification.
    Disable the real-time results and plots to avoid any GUI issues.

    Usage: 
    tests_sports2d
        OR
    python tests.py
'''

## INIT
from importlib.metadata import version
import json
import os
import toml
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = version("sports2d")
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def test_wrap_angle_series_to_principal_preserves_nans_and_bounds_values():
    '''
    Verify angle wrapping keeps NaNs and maps values to principal range.
    '''

    from Sports2D.Utilities.common import wrap_angle_series_to_principal

    values = np.array([np.nan, -540.0, -190.0, -181.0, -180.0, -179.0, 0.0, 179.0, 180.0, 181.0, 360.0, 402.59])
    wrapped = wrap_angle_series_to_principal(values)
    expected = np.array([np.nan, -180.0, 170.0, 179.0, -180.0, -179.0, 0.0, 179.0, -180.0, -179.0, 0.0, 42.59])

    assert np.isnan(wrapped[0])
    assert np.allclose(wrapped[1:], expected[1:], atol=1e-9)


def test_wrap_angle_series_to_principal_output_range():
    '''
    Verify wrapped values remain inside [-180, 180] for finite entries.
    '''

    from Sports2D.Utilities.common import wrap_angle_series_to_principal

    values = np.array([-721.0, -360.0, -181.0, -180.0, -90.0, 0.0, 90.0, 179.0, 180.0, 181.0, 359.0, 540.0, np.nan])
    wrapped = wrap_angle_series_to_principal(values)
    valid = wrapped[~np.isnan(wrapped)]

    assert np.all(valid >= -180.0)
    assert np.all(valid <= 180.0)


def test_extract_ball_centers_parses_xyxy_boxes():
    '''
    Verify ball center extraction from detector xyxy boxes.
    '''

    from Sports2D.process import extract_ball_centers

    detection_meta = {
        'ball_boxes': np.array([
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 140.0, 260.0],
        ])
    }
    centers = extract_ball_centers(detection_meta)

    assert centers == [(20, 30), (120, 230)]


def test_select_ball_center_applies_nearest_and_jump_gate():
    '''
    Verify ball center selection favors continuity and can reject large jumps.
    '''

    from Sports2D.process import select_ball_center

    candidates = [(100, 100), (220, 220), (103, 98)]
    chosen = select_ball_center(candidates, previous_center=(101, 101), max_jump_px=20)
    rejected = select_ball_center(candidates, previous_center=(0, 0), max_jump_px=10)

    assert chosen == (103, 98)
    assert rejected is None


def test_select_ball_center_uses_switch_hysteresis_with_velocity():
    '''
    Verify ball center selection avoids unstable switching between nearby candidates.
    '''

    from Sports2D.process import select_ball_center

    candidates = [(104, 100), (112, 100)]
    sticky = select_ball_center(
        candidates,
        previous_center=(100, 100),
        previous_velocity=(10.0, 0.0),
        max_jump_px=40,
        switch_margin_px=8.0,
    )
    switching = select_ball_center(
        candidates,
        previous_center=(100, 100),
        previous_velocity=(10.0, 0.0),
        max_jump_px=40,
        switch_margin_px=0.0,
    )

    assert sticky == (104, 100)
    assert switching == (112, 100)


def test_select_ball_center_keeps_same_frame_duplicate_candidates_stable():
    '''
    Verify duplicate same-frame ball candidates do not destabilize the tracked center.
    '''

    from Sports2D.process import select_ball_center

    selected = select_ball_center(
        [(120, 80), (120, 80), (148, 112)],
        previous_center=(118, 79),
        max_jump_px=40,
    )

    assert selected == (120, 80)


def test_dedupe_ball_detections_keeps_highest_score_duplicate():
    '''
    Verify same-frame duplicate ball boxes collapse to one candidate before tracking.
    '''

    from Sports2D.process import dedupe_ball_detections

    deduped_boxes, deduped_scores = dedupe_ball_detections(
        np.array([
            [40.0, 50.0, 52.0, 62.0],
            [41.0, 50.0, 53.0, 62.0],
            [90.0, 90.0, 104.0, 104.0],
        ], dtype=np.float32),
        np.array([0.42, 0.91, 0.35], dtype=np.float32),
    )

    assert deduped_boxes.shape == (2, 4)
    assert deduped_scores.tolist() == pytest.approx([0.91, 0.35])
    assert deduped_boxes[0].tolist() == pytest.approx([41.0, 50.0, 53.0, 62.0])


def test_track_balls_sports2d_keeps_ids_when_detection_order_swaps():
    '''
    Verify sports2d ball tracking preserves IDs even when detector output order flips.
    '''

    from Sports2D.process import track_balls_sports2d

    boxes_f1 = np.array([
        [10.0, 10.0, 20.0, 20.0],
        [100.0, 100.0, 112.0, 112.0],
    ], dtype=np.float32)

    tracks_f1, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        boxes_f1,
        previous_keypoints=np.empty((0, 1, 2), dtype=np.float32),
        previous_track_ids=[],
        previous_missing_counts=[],
        next_track_id=0,
        max_dist=80.0,
        max_missing_frames=3,
    )

    left_id = next(track['id'] for track in tracks_f1 if track['center'][0] < 60)
    right_id = next(track['id'] for track in tracks_f1 if track['center'][0] >= 60)

    # Same two balls, reversed detector ordering in next frame.
    boxes_f2 = np.array([
        [102.0, 102.0, 114.0, 114.0],
        [12.0, 12.0, 22.0, 22.0],
    ], dtype=np.float32)
    tracks_f2, _, _, _, next_id = track_balls_sports2d(
        boxes_f2,
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        max_dist=80.0,
        max_missing_frames=3,
    )

    left_id_after = next(track['id'] for track in tracks_f2 if track['center'][0] < 60)
    right_id_after = next(track['id'] for track in tracks_f2 if track['center'][0] >= 60)

    assert left_id_after == left_id
    assert right_id_after == right_id
    assert next_id == 2


def test_track_balls_sports2d_reuses_id_after_short_missing_gap():
    '''
    Verify a temporarily missing ball keeps the same ID within missing-frame budget.
    '''

    from Sports2D.process import track_balls_sports2d

    tracks_f1, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        np.array([[50.0, 50.0, 62.0, 62.0]], dtype=np.float32),
        previous_keypoints=np.empty((0, 1, 2), dtype=np.float32),
        previous_track_ids=[],
        previous_missing_counts=[],
        next_track_id=0,
        max_dist=50.0,
        max_missing_frames=2,
    )
    assert tracks_f1[0]['id'] == 0

    tracks_f2, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        np.empty((0, 4), dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        max_dist=50.0,
        max_missing_frames=2,
    )
    assert tracks_f2[0]['id'] == 0
    assert tracks_f2[0]['visible'] is False

    tracks_f3, _, _, _, next_id = track_balls_sports2d(
        np.array([[52.0, 51.0, 64.0, 63.0]], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        max_dist=50.0,
        max_missing_frames=2,
    )
    assert tracks_f3[0]['id'] == 0
    assert tracks_f3[0]['visible'] is True
    assert next_id == 1


def test_track_balls_sports2d_reassociates_after_three_missing_frames():
    '''
    Verify a ball track survives a three-frame gap and re-associates to the same ID.
    '''

    from Sports2D.process import track_balls_sports2d

    empty_boxes = np.empty((0, 4), dtype=np.float32)
    prev_kpts = np.empty((0, 1, 2), dtype=np.float32)
    prev_ids = []
    prev_missing = []
    next_id = 0

    tracks_f1, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        np.array([[50.0, 50.0, 62.0, 62.0]], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        max_dist=50.0,
        max_missing_frames=3,
    )
    assert tracks_f1[0]['id'] == 0

    for expected_missing in (1, 2, 3):
        tracks_f1, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
            empty_boxes,
            previous_keypoints=prev_kpts,
            previous_track_ids=prev_ids,
            previous_missing_counts=prev_missing,
            next_track_id=next_id,
            max_dist=50.0,
            max_missing_frames=3,
        )
        assert tracks_f1[0]['id'] == 0
        assert tracks_f1[0]['visible'] is False
        assert tracks_f1[0]['missing'] == expected_missing

    tracks_f2, _, _, _, next_id = track_balls_sports2d(
        np.array([[52.0, 51.0, 64.0, 63.0]], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        max_dist=50.0,
        max_missing_frames=3,
    )

    assert tracks_f2[0]['id'] == 0
    assert tracks_f2[0]['visible'] is True
    assert tracks_f2[0]['missing'] == 0
    assert next_id == 1


def test_track_balls_sports2d_uses_velocity_state_to_reassociate_after_gap():
    '''
    Verify motion-aware reassociation can recover the same ID when raw last-center distance is too large.
    '''

    from Sports2D.process import track_balls_sports2d

    velocity_state = {}
    prev_kpts = np.empty((0, 1, 2), dtype=np.float32)
    prev_ids = []
    prev_missing = []
    next_id = 0

    tracks_f1, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        track_velocities_by_id=velocity_state,
        max_dist=18.0,
        max_missing_frames=3,
    )
    assert tracks_f1[0]['id'] == 0

    tracks_f2, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        np.array([[30.0, 10.0, 40.0, 20.0]], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        track_velocities_by_id=velocity_state,
        max_dist=25.0,
        max_missing_frames=3,
    )
    assert tracks_f2[0]['id'] == 0

    for _ in range(2):
        _, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
            np.empty((0, 4), dtype=np.float32),
            previous_keypoints=prev_kpts,
            previous_track_ids=prev_ids,
            previous_missing_counts=prev_missing,
            next_track_id=next_id,
            track_velocities_by_id=velocity_state,
            max_dist=25.0,
            max_missing_frames=3,
        )

    tracks_f5, _, _, _, next_id = track_balls_sports2d(
        np.array([[90.0, 10.0, 100.0, 20.0]], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        track_velocities_by_id=velocity_state,
        max_dist=25.0,
        max_missing_frames=3,
    )

    assert tracks_f5[0]['id'] == 0
    assert tracks_f5[0]['visible'] is True
    assert next_id == 1


def test_track_balls_sports2d_initializes_new_track_velocity_without_nan():
    '''
    Verify unmatched new raw tracks start with a finite zero velocity estimate.
    '''

    from Sports2D.process import track_balls_sports2d

    velocity_state = {}
    tracks_f1, prev_kpts, prev_ids, prev_missing, next_id = track_balls_sports2d(
        np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
        previous_keypoints=np.empty((0, 1, 2), dtype=np.float32),
        previous_track_ids=[],
        previous_missing_counts=[],
        next_track_id=0,
        track_velocities_by_id=velocity_state,
        max_dist=30.0,
        max_missing_frames=2,
    )

    tracks_f2, _, _, _, _ = track_balls_sports2d(
        np.array([
            [12.0, 10.0, 22.0, 20.0],
            [100.0, 100.0, 112.0, 112.0],
        ], dtype=np.float32),
        previous_keypoints=prev_kpts,
        previous_track_ids=prev_ids,
        previous_missing_counts=prev_missing,
        next_track_id=next_id,
        track_velocities_by_id=velocity_state,
        max_dist=30.0,
        max_missing_frames=2,
    )

    new_track_id = max(track['id'] for track in tracks_f2)
    assert velocity_state[new_track_id] == pytest.approx((0.0, 0.0))
    assert np.isfinite(np.asarray(velocity_state[new_track_id], dtype=float)).all()


def test_parse_ball_ordering_method_falls_back_for_invalid_values():
    '''
    Verify invalid ball ordering methods fall back to the provided default.
    '''

    from Sports2D.process import _parse_ball_ordering_method

    assert _parse_ball_ordering_method('largest_size') == 'largest_size'
    assert _parse_ball_ordering_method('not_a_mode', default='first_detected') == 'first_detected'


def test_parse_ball_detector_backend_accepts_matching_detector_alias():
    '''
    Verify ball detector backend parser treats the same detector name as 'same'.
    '''

    from Sports2D.process import _parse_ball_detector_backend

    assert _parse_ball_detector_backend('same', synthpose_detector='yolox') == 'same'
    assert _parse_ball_detector_backend('sam3', synthpose_detector='yolox') == 'sam3'
    assert _parse_ball_detector_backend('yolox', synthpose_detector='yolox') == 'same'
    assert _parse_ball_detector_backend('yolo26', synthpose_detector='yolo26') == 'same'
    assert _parse_ball_detector_backend('not_a_backend', synthpose_detector='yolox') == 'same'


def test_synthpose_tracker_normalizes_matching_ball_detector_alias():
    '''
    Verify SynthPose tracker backend normalization accepts the active detector name as an alias for 'same'.
    '''

    from Sports2D.Utilities.synthpose_tracker import _normalize_ball_detector_backend

    assert _normalize_ball_detector_backend('same', detector='yolox') == 'same'
    assert _normalize_ball_detector_backend('sam3', detector='yolox') == 'sam3'
    assert _normalize_ball_detector_backend('yolox', detector='yolox') == 'same'
    assert _normalize_ball_detector_backend('yolo26', detector='yolo26') == 'same'
    assert _normalize_ball_detector_backend('not_a_backend', detector='yolox') == 'same'


def test_normalize_synthpose_detector_accepts_yolo26_and_rejects_unknown():
    '''
    Verify supported SynthPose detectors include YOLO26 and reject unknown values.
    '''

    from Sports2D.Utilities.synthpose_tracker import _normalize_synthpose_detector

    assert _normalize_synthpose_detector('yolo26') == 'yolo26'
    with pytest.raises(ValueError, match='Unsupported synthpose_detector'):
        _normalize_synthpose_detector('not_a_detector')


def test_synthpose_tracker_resolves_yolo26_detector_sizes():
    '''
    Verify YOLO26 detector sizing supports the nano variant for lightweight mode.
    '''

    from Sports2D.Utilities.synthpose_tracker import SynthPosePoseTracker

    assert SynthPosePoseTracker._resolve_detector_size('lightweight', detector='yolo26') == 'n'
    assert SynthPosePoseTracker._resolve_detector_size('n', detector='yolo26') == 'n'
    assert SynthPosePoseTracker._resolve_detector_size('tiny', detector='yolo26') == 'n'


def test_parse_video_codec_accepts_aliases_and_rejects_invalid_values():
    '''
    Verify video codec parser normalizes aliases and rejects unsupported codecs.
    '''

    from Sports2D.process import _parse_video_codec

    assert _parse_video_codec('mp4v') == 'mp4v'
    assert _parse_video_codec('mpeg4') == 'mp4v'
    assert _parse_video_codec('h264') == 'h264'
    assert _parse_video_codec('avc1') == 'h264'
    with pytest.raises(ValueError, match='Invalid video_codec'):
        _parse_video_codec('vp9')


def test_remap_pose_model_ids_by_keypoint_names_uses_tensor_order():
    '''
    Verify saved-overlay pose-model remap follows keypoint tensor order, not tree order.
    '''

    from anytree import Node, PreOrderIter
    from Sports2D.process import _remap_pose_model_ids_by_keypoint_names

    root = Node('root', id=None)
    hip = Node('Hip', parent=root, id=10)
    Node('RKnee', parent=hip, id=4)
    Node('RAnkle', parent=hip, id=6)
    Node('LWrist', parent=root, id=2)
    Node('UnknownMarker', parent=root, id=99)

    remapped = _remap_pose_model_ids_by_keypoint_names(
        root,
        ['LWrist', 'RKnee', 'Hip'],
    )
    remapped_ids = {
        node.name: node.id
        for node in PreOrderIter(remapped)
    }
    original_ids = {
        node.name: node.id
        for node in PreOrderIter(root)
    }

    assert remapped_ids['Hip'] == 2
    assert remapped_ids['RKnee'] == 1
    assert remapped_ids['LWrist'] == 0
    assert remapped_ids['RAnkle'] is None
    assert remapped_ids['UnknownMarker'] is None
    assert original_ids['Hip'] == 10
    assert original_ids['RKnee'] == 4
    assert original_ids['LWrist'] == 2


def test_build_ball_click_pose_coords_tracks_first_appearance_order():
    '''
    Verify ball click UI coordinates preserve track first-appearance order and bbox mapping.
    '''

    from Sports2D.process import _build_ball_click_pose_coords

    all_frames_ball_tracks = [
        [
            {'id': 7, 'box': np.array([10.0, 20.0, 30.0, 40.0]), 'visible': True},
            {'id': 2, 'box': np.array([50.0, 60.0, 70.0, 80.0]), 'visible': True},
        ],
        [
            {'id': 2, 'box': np.array([52.0, 62.0, 72.0, 82.0]), 'visible': True},
            {'id': 9, 'box': np.array([5.0, 6.0, 8.0, 9.0]), 'visible': True},
        ],
    ]

    coords, track_ids = _build_ball_click_pose_coords(all_frames_ball_tracks)

    assert track_ids == [7, 2, 9]
    assert coords.shape == (2, 3, 2, 2)
    assert np.allclose(coords[0, 0, 0], [10.0, 20.0])
    assert np.allclose(coords[0, 0, 1], [30.0, 40.0])
    assert np.allclose(coords[1, 1, 0], [52.0, 62.0])
    assert np.allclose(coords[1, 1, 1], [72.0, 82.0])
    assert np.allclose(coords[1, 2, 0], [5.0, 6.0])
    assert np.allclose(coords[1, 2, 1], [8.0, 9.0])
    assert np.isnan(coords[0, 2]).all()


def test_select_ball_track_id_auto_orders_by_first_and_last_detected():
    '''
    Verify first_detected/last_detected ordering uses first-seen frame stats.
    '''

    from Sports2D.process import select_ball_track_id

    tracked_balls = [
        {'id': 2, 'center': (110, 40), 'box': np.array([100, 30, 120, 50]), 'score': 0.6, 'visible': True, 'missing': 0},
        {'id': 5, 'center': (30, 40), 'box': np.array([20, 30, 40, 50]), 'score': 0.7, 'visible': True, 'missing': 0},
    ]
    track_stats = {
        2: {'first_seen_frame': 14, 'area_sum': 400.0, 'area_count': 1, 'score_sum': 0.6, 'score_count': 1, 'displacement_sum': 5.0},
        5: {'first_seen_frame': 3, 'area_sum': 400.0, 'area_count': 1, 'score_sum': 0.7, 'score_count': 1, 'displacement_sum': 2.0},
    }

    first_id, first_center = select_ball_track_id(
        tracked_balls,
        selection_mode='auto',
        previous_selected_id=None,
        ordering_method='first_detected',
        track_stats_by_id=track_stats,
    )
    last_id, last_center = select_ball_track_id(
        tracked_balls,
        selection_mode='auto',
        previous_selected_id=None,
        ordering_method='last_detected',
        track_stats_by_id=track_stats,
    )

    assert first_id == 5
    assert first_center == (30, 40)
    assert last_id == 2
    assert last_center == (110, 40)


def test_select_ball_track_id_on_click_uses_first_detected_until_manual_pick():
    '''
    Verify on_click ordering behaves like first_detected before manual click selection is applied.
    '''

    from Sports2D.process import select_ball_track_id

    tracked_balls = [
        {'id': 3, 'center': (90, 40), 'box': np.array([80, 30, 100, 50]), 'score': 0.6, 'visible': True, 'missing': 0},
        {'id': 8, 'center': (20, 40), 'box': np.array([10, 30, 30, 50]), 'score': 0.7, 'visible': True, 'missing': 0},
    ]
    track_stats = {
        3: {'first_seen_frame': 20, 'area_sum': 400.0, 'area_count': 1, 'score_sum': 0.6, 'score_count': 1, 'displacement_sum': 3.0},
        8: {'first_seen_frame': 4, 'area_sum': 400.0, 'area_count': 1, 'score_sum': 0.7, 'score_count': 1, 'displacement_sum': 2.0},
    }

    selected_id, selected_center = select_ball_track_id(
        tracked_balls,
        selection_mode='auto',
        previous_selected_id=None,
        ordering_method='on_click',
        track_stats_by_id=track_stats,
    )

    assert selected_id == 8
    assert selected_center == (20, 40)


def test_select_ball_track_id_auto_orders_by_size_displacement_and_likelihood():
    '''
    Verify size/displacement/likelihood ordering modes choose the expected track ID.
    '''

    from Sports2D.process import select_ball_track_id

    tracked_balls = [
        {'id': 1, 'center': (20, 20), 'box': np.array([15, 15, 25, 25]), 'score': 0.4, 'visible': True, 'missing': 0},
        {'id': 2, 'center': (60, 20), 'box': np.array([50, 10, 80, 30]), 'score': 0.9, 'visible': True, 'missing': 0},
        {'id': 3, 'center': (100, 20), 'box': np.array([96, 16, 104, 24]), 'score': 0.7, 'visible': True, 'missing': 0},
    ]
    track_stats = {
        1: {'first_seen_frame': 1, 'area_sum': 100.0, 'area_count': 1, 'score_sum': 0.4, 'score_count': 1, 'displacement_sum': 10.0},
        2: {'first_seen_frame': 2, 'area_sum': 600.0, 'area_count': 1, 'score_sum': 0.9, 'score_count': 1, 'displacement_sum': 2.0},
        3: {'first_seen_frame': 3, 'area_sum': 64.0, 'area_count': 1, 'score_sum': 0.7, 'score_count': 1, 'displacement_sum': 20.0},
    }

    largest_id, _ = select_ball_track_id(
        tracked_balls, selection_mode='auto', ordering_method='largest_size',
        track_stats_by_id=track_stats, previous_selected_id=None,
    )
    smallest_id, _ = select_ball_track_id(
        tracked_balls, selection_mode='auto', ordering_method='smallest_size',
        track_stats_by_id=track_stats, previous_selected_id=None,
    )
    greatest_disp_id, _ = select_ball_track_id(
        tracked_balls, selection_mode='auto', ordering_method='greatest_displacement',
        track_stats_by_id=track_stats, previous_selected_id=None,
    )
    least_disp_id, _ = select_ball_track_id(
        tracked_balls, selection_mode='auto', ordering_method='least_displacement',
        track_stats_by_id=track_stats, previous_selected_id=None,
    )
    highest_like_id, _ = select_ball_track_id(
        tracked_balls, selection_mode='auto', ordering_method='highest_likelihood',
        track_stats_by_id=track_stats, previous_selected_id=None,
    )

    assert largest_id == 2
    assert smallest_id == 3
    assert greatest_disp_id == 3
    assert least_disp_id == 2
    assert highest_like_id == 2


def test_select_ball_track_id_highest_likelihood_falls_back_when_scores_missing():
    '''
    Verify highest_likelihood falls back deterministically when no score stats exist.
    '''

    from Sports2D.process import select_ball_track_id

    tracked_balls = [
        {'id': 4, 'center': (20, 10), 'box': np.array([15, 5, 25, 15]), 'score': np.nan, 'visible': True, 'missing': 0},
        {'id': 9, 'center': (60, 10), 'box': np.array([55, 5, 65, 15]), 'score': np.nan, 'visible': True, 'missing': 0},
    ]
    track_stats = {
        4: {'first_seen_frame': 1, 'area_sum': 100.0, 'area_count': 1, 'score_sum': 0.0, 'score_count': 0, 'displacement_sum': 0.0},
        9: {'first_seen_frame': 7, 'area_sum': 100.0, 'area_count': 1, 'score_sum': 0.0, 'score_count': 0, 'displacement_sum': 0.0},
    }

    selected_id, selected_center = select_ball_track_id(
        tracked_balls,
        selection_mode='auto',
        previous_selected_id=None,
        ordering_method='highest_likelihood',
        track_stats_by_id=track_stats,
    )

    assert selected_id == 4
    assert selected_center == (20, 10)


def test_select_ball_track_id_supports_explicit_id_mode():
    '''
    Verify explicit ID mode tracks only the requested ball ID.
    '''

    from Sports2D.process import select_ball_track_id

    tracked_balls = [
        {'id': 3, 'center': (40, 40), 'box': np.array([35, 35, 45, 45]), 'visible': True, 'missing': 0},
        {'id': 7, 'center': (90, 90), 'box': np.array([84, 84, 96, 96]), 'visible': True, 'missing': 0},
    ]

    selected_id, selected_center = select_ball_track_id(
        tracked_balls,
        selection_mode='id',
        requested_track_id=7,
        previous_selected_id=None,
        previous_selected_center=None,
    )
    missing_id, missing_center = select_ball_track_id(
        tracked_balls,
        selection_mode='id',
        requested_track_id=11,
        previous_selected_id=None,
        previous_selected_center=None,
    )

    assert selected_id == 7
    assert selected_center == (90, 90)
    assert missing_id == 11
    assert missing_center is None


def test_select_ball_track_id_preserves_selected_id_across_raw_split():
    '''
    Verify selected-ball continuity keeps the prior selected ID when a new raw fragment continues the motion.
    '''

    from Sports2D.process import select_ball_track_id

    tracked_balls = [
        {'id': 19, 'center': (98, 42), 'box': np.array([92, 36, 104, 48]), 'score': 0.81, 'visible': True, 'missing': 0},
    ]

    selected_id, selected_center = select_ball_track_id(
        tracked_balls,
        selection_mode='auto',
        previous_selected_id=7,
        previous_selected_center=(90, 40),
        previous_selected_velocity=(8.0, 2.0),
        ordering_method='first_detected',
        track_stats_by_id={19: {'first_seen_frame': 12, 'score_sum': 0.81, 'score_count': 1}},
        max_recovery_dist=24.0,
    )

    assert selected_id == 7
    assert selected_center == (98, 42)


def test_sam3_prompt_presets_resolve_ball_and_broad_jump_targets():
    '''
    Verify SAM3 target presets normalize aliases and return the expected prompts.
    '''

    from Sports2D.Utilities.sam3_detector import (
        normalize_sam3_target,
        resolve_sam3_prompts,
    )

    assert normalize_sam3_target('ball') == 'ball'
    assert normalize_sam3_target('broad jump') == 'broad_jump'
    assert resolve_sam3_prompts('ball') == ['person', 'sports ball']
    assert resolve_sam3_prompts('broad_jump') == ['person']


def test_extract_prompt_instances_derives_boxes_from_masks():
    '''
    Verify SAM3 prompt results can recover xyxy boxes from binary masks.
    '''

    from Sports2D.Utilities.sam3_detector import extract_prompt_instances

    masks = np.zeros((2, 6, 7), dtype=np.uint8)
    masks[0, 1:4, 2:5] = 1
    masks[1, 4:6, 0:2] = 1

    boxes, scores, masks_list = extract_prompt_instances({
        'masks': masks,
        'scores': np.array([0.9, 0.4], dtype=np.float32),
    })

    assert np.allclose(boxes, np.array([[2, 1, 4, 3], [0, 4, 1, 5]], dtype=np.float32))
    assert np.allclose(scores, np.array([0.9, 0.4], dtype=np.float32))
    assert len(masks_list) == 2


def test_build_sam3_detection_metadata_maps_person_and_ball_prompts():
    '''
    Verify SAM3 prompt metadata maps person/ball prompts onto Sports2D class IDs.
    '''

    from Sports2D.Utilities.sam3_detector import (
        PERSON_CLASS_ID,
        SPORTS_BALL_CLASS_ID,
        build_sam3_detection_metadata,
    )

    prompts = ['person', 'a ball being thrown by a person']
    metadata = build_sam3_detection_metadata(
        boxes=np.array([
            [10.0, 15.0, 40.0, 80.0],
            [100.0, 110.0, 112.0, 124.0],
        ], dtype=np.float32),
        scores=np.array([0.95, 0.72], dtype=np.float32),
        prompts=prompts,
        prompt_indices=np.array([0, 1], dtype=np.int32),
    )

    assert metadata['classes'].tolist() == [PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID]
    assert metadata['class_names'].tolist() == prompts
    assert metadata['person_boxes'].shape == (1, 4)
    assert metadata['ball_boxes'].shape == (1, 4)
    assert np.allclose(metadata['ball_scores'], np.array([0.72], dtype=np.float32))


def test_draw_sam3_mask_overlay_noops_without_masks():
    '''
    Verify SAM3 realtime overlay leaves the frame unchanged when masks are absent.
    '''

    from Sports2D.process import draw_sam3_mask_overlay

    image = np.zeros((6, 7, 3), dtype=np.uint8)
    output = draw_sam3_mask_overlay(
        image.copy(),
        detection_meta={'classes': np.array([0], dtype=np.int32)},
        alpha=0.4,
    )

    assert np.array_equal(output, image)


def test_draw_sam3_mask_overlay_blends_person_and_ball_masks():
    '''
    Verify SAM3 realtime overlay colors person and ball masks without touching other pixels.
    '''

    from Sports2D.process import draw_sam3_mask_overlay

    image = np.zeros((8, 9, 3), dtype=np.uint8)
    person_mask = np.zeros((8, 9), dtype=np.uint8)
    ball_mask = np.zeros((8, 9), dtype=np.uint8)
    person_mask[1:4, 1:4] = 1
    ball_mask[5:7, 6:8] = 1

    output = draw_sam3_mask_overlay(
        image.copy(),
        detection_meta={
            'classes': np.array([0, 32], dtype=np.int32),
            'masks': [person_mask, ball_mask],
        },
        alpha=0.5,
        person_color=(20, 120, 200),
        ball_color=(0, 180, 255),
    )

    assert np.any(output[person_mask.astype(bool)] != 0)
    assert np.any(output[ball_mask.astype(bool)] != 0)
    assert np.array_equal(output[0, 0], np.array([0, 0, 0], dtype=np.uint8))


def test_filter_sam3_detection_meta_classes_keeps_only_ball_masks():
    '''
    Verify SAM3 metadata filtering can keep only ball-class masks for export overlays.
    '''

    from Sports2D.process import filter_sam3_detection_meta_classes
    from Sports2D.Utilities.sam3_detector import PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID

    detection_meta = {
        'boxes': np.array([[1, 1, 4, 4], [5, 5, 7, 7]], dtype=np.float32),
        'classes': np.array([PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID], dtype=np.int32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'class_names': np.array(['person', 'sports ball'], dtype=object),
        'prompt_indices': np.array([0, 1], dtype=np.int32),
        'masks': [
            np.array([[1, 1], [1, 1]], dtype=np.uint8),
            np.array([[0, 1], [1, 1]], dtype=np.uint8),
        ],
    }

    filtered = filter_sam3_detection_meta_classes(
        detection_meta,
        allowed_class_ids=[SPORTS_BALL_CLASS_ID],
    )

    assert filtered['classes'].tolist() == [SPORTS_BALL_CLASS_ID]
    assert filtered['ball_boxes'].shape == (1, 4)
    assert filtered['person_boxes'].shape == (0, 4)
    assert len(filtered['masks']) == 1
    assert filtered['class_names'].tolist() == ['sports ball']


def test_build_ball_export_series_uses_selected_ball_metadata():
    '''
    Verify per-frame ball export records keep selected track metadata and visibility.
    '''

    from Sports2D.process import build_ball_export_series

    export_series = build_ball_export_series(
        pd.Series([0.0, 0.5], name='time'),
        all_frames_ball_centers=[(14, 16), None],
        all_frames_ball_boxes=[
            np.array([[10.0, 12.0, 18.0, 20.0]], dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        ],
        all_frames_ball_scores=[
            np.array([0.88], dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ],
        all_frames_ball_tracks=[
            [{'id': 7, 'center': (14, 16), 'box': np.array([10.0, 12.0, 18.0, 20.0], dtype=np.float32), 'score': 0.88, 'visible': True}],
            [{'id': 7, 'center': (14, 16), 'box': None, 'score': float('nan'), 'visible': False}],
        ],
        all_frames_selected_ball_ids=[7, 7],
        all_frames_sam3_ball_mask_meta=[{'masks': [np.ones((2, 2), dtype=np.uint8)]}, {}],
        frame_offset=12,
        multi_id_tracking=True,
    )

    assert export_series[0]['frame_index'] == 12
    assert export_series[0]['ball']['visible'] is True
    assert export_series[0]['ball']['track_id'] == 7
    assert export_series[0]['ball']['source_track_id'] == 7
    assert export_series[0]['ball']['center_xy'] == [14, 16]
    assert export_series[0]['ball']['score'] == pytest.approx(0.88)
    assert export_series[0]['ball']['mask_available'] is True
    assert export_series[1]['ball']['visible'] is False
    assert export_series[1]['ball']['track_id'] == 7
    assert export_series[1]['ball']['source_track_id'] is None
    assert export_series[1]['ball']['center_xy'] is None
    assert export_series[1]['ball']['box_xyxy'] is None
    assert export_series[1]['ball']['score'] is None
    assert export_series[1]['ball']['ball_keypoints_2d'] == [None, None, None]


def test_build_ball_export_series_keeps_selected_track_id_through_raw_split():
    '''
    Verify export timelines stay anchored to the selected ball ID when raw track IDs split.
    '''

    from Sports2D.process import build_ball_export_series

    export_series = build_ball_export_series(
        pd.Series([0.0, 0.5], name='time'),
        all_frames_ball_centers=[(14, 16), (18, 20)],
        all_frames_ball_boxes=[
            np.array([[10.0, 12.0, 18.0, 20.0]], dtype=np.float32),
            np.array([[16.0, 18.0, 20.0, 22.0]], dtype=np.float32),
        ],
        all_frames_ball_scores=[
            np.array([0.88], dtype=np.float32),
            np.array([0.77], dtype=np.float32),
        ],
        all_frames_ball_tracks=[
            [
                {
                    'id': 7,
                    'center': (14, 16),
                    'box': np.array([10.0, 12.0, 18.0, 20.0], dtype=np.float32),
                    'score': 0.88,
                    'visible': True,
                },
            ],
            [
                {
                    'id': 19,
                    'center': (18, 20),
                    'box': np.array([16.0, 18.0, 20.0, 22.0], dtype=np.float32),
                    'score': 0.77,
                    'visible': True,
                },
            ],
        ],
        all_frames_selected_ball_ids=[7, 7],
        all_frames_sam3_ball_mask_meta=[{'masks': [np.ones((2, 2), dtype=np.uint8)]}, {}],
        multi_id_tracking=True,
    )

    assert export_series[0]['ball']['track_id'] == 7
    assert export_series[0]['ball']['source_track_id'] == 7
    assert export_series[0]['ball']['visible'] is True
    assert export_series[1]['ball']['track_id'] == 7
    assert export_series[1]['ball']['source_track_id'] == 19
    assert export_series[1]['ball']['visible'] is True
    assert export_series[1]['ball']['center_xy'] == [18, 20]
    assert export_series[1]['ball']['box_xyxy'] == [16.0, 18.0, 20.0, 22.0]
    assert export_series[1]['ball']['score'] == pytest.approx(0.77)


def test_stitch_selected_ball_timeline_recovers_earlier_fragment_from_clicked_seed():
    '''
    Verify on-click stitching can recover an earlier compatible fragment before the clicked raw ID appears.
    '''

    from Sports2D.process import stitch_selected_ball_timeline

    stitched_ids, stitched_centers = stitch_selected_ball_timeline(
        [
            [{'id': 7, 'center': (20, 20), 'box': np.array([14, 14, 26, 26]), 'score': 0.7, 'visible': True, 'missing': 0}],
            [{'id': 19, 'center': (30, 20), 'box': np.array([24, 14, 36, 26]), 'score': 0.8, 'visible': True, 'missing': 0}],
            [{'id': 19, 'center': (40, 20), 'box': np.array([34, 14, 46, 26]), 'score': 0.9, 'visible': True, 'missing': 0}],
        ],
        selected_track_id=19,
        max_jump_px=20.0,
    )

    assert stitched_ids == [19, 19, 19]
    assert stitched_centers == [(20, 20), (30, 20), (40, 20)]


def test_stitch_selected_ball_timeline_rejects_far_fragment_when_jump_is_unbounded():
    '''
    Verify on-click stitching still uses a conservative continuity gate when max_jump_px is disabled.
    '''

    from Sports2D.process import stitch_selected_ball_timeline

    stitched_ids, stitched_centers = stitch_selected_ball_timeline(
        [
            [{'id': 7, 'center': (20, 20), 'box': np.array([14, 14, 26, 26]), 'score': 0.7, 'visible': True, 'missing': 0}],
            [{'id': 19, 'center': (400, 20), 'box': np.array([394, 14, 406, 26]), 'score': 0.8, 'visible': True, 'missing': 0}],
        ],
        selected_track_id=19,
        max_jump_px=None,
    )

    assert stitched_ids == [None, 19]
    assert stitched_centers == [None, (400, 20)]


def test_write_ball_pose_json_writes_frame_payloads(tmp_path):
    '''
    Verify pose_ball JSON export writes stable per-frame files.
    '''

    from Sports2D.process import write_ball_pose_json

    output_dir = tmp_path / 'pose_ball'
    series = [
        {
            'frame_index': 5,
            'time': 0.2,
            'ball': {
                'visible': True,
                'track_id': None,
                'source_track_id': None,
                'score': 0.91,
                'center_xy': [20, 30],
                'box_xyxy': [10.0, 20.0, 30.0, 40.0],
                'ball_keypoints_2d': [20.0, 30.0, 0.91],
                'mask_available': False,
            },
        },
    ]

    write_ball_pose_json(series, output_dir, 'demo_output')

    payload = json.loads((output_dir / 'demo_output_000005.json').read_text(encoding='utf-8'))
    assert payload['frame_index'] == 5
    assert payload['time'] == pytest.approx(0.2)
    assert payload['balls'][0]['center_xy'] == [20, 30]
    assert payload['balls'][0]['ball_keypoints_2d'] == [20.0, 30.0, 0.91]


def test_write_ball_blender_helper_writes_follow_marker_script(tmp_path):
    '''
    Verify Blender helper export writes a marker-following sphere script.
    '''

    from Sports2D.Utilities.ball_blender import write_ball_blender_helper

    script_path = write_ball_blender_helper(tmp_path, 'demo_output')

    assert script_path.name == 'demo_output_ball_mesh_blender.py'
    content = script_path.read_text(encoding='utf-8')
    assert 'demo_output_m_person00.trc' in content
    assert 'BALL_RADIUS_M' in content
    assert 'COPY_LOCATION' in content
    assert 'select the ball marker object' in content.lower()


def test_append_ball_marker_to_trc_data_adds_trailing_triplet():
    '''
    Verify TRC exports can append a trailing ball marker without mutating prior markers.
    '''

    from Sports2D.process import append_ball_marker_to_trc_data, build_ball_trc_data

    trc_data = pd.DataFrame(
        {
            'time': [0.0, 0.5],
            'Hip': [1.0, 2.0],
            'Hip.1': [3.0, 4.0],
            'Hip.2': [0.0, 0.0],
        }
    )
    trc_data.columns = ['time', 'Hip', 'Hip', 'Hip']
    ball_trc = build_ball_trc_data(
        [
            {'ball': {'center_xy': [10, 20]}},
            {'ball': {'center_xy': None}},
        ],
        index=trc_data.index,
        marker_name='ball',
    )

    merged = append_ball_marker_to_trc_data(trc_data, ball_trc, marker_name='ball')

    assert merged.columns.tolist() == ['time', 'Hip', 'Hip', 'Hip', 'ball', 'ball', 'ball']
    assert merged.iloc[0, -3:].tolist() == pytest.approx([10.0, 20.0, 0.0])
    assert np.isnan(merged.iloc[1, -3])
    assert np.isnan(merged.iloc[1, -2])
    assert np.isnan(merged.iloc[1, -1])


def test_append_trc_marker_aliases_adds_small_toe_triplets_for_synthpose():
    '''
    Verify SynthPose metatarsal markers are exported under small-toe aliases.
    '''

    from Sports2D.process import append_trc_marker_aliases
    from Sports2D.Utilities.synthpose_skeleton import SYNTHPOSE_MARKER_ALIASES

    trc_data = pd.DataFrame(
        {
            'time': [0.0, 0.5],
            'R5Meta': [1.0, 2.0],
            'R5Meta.1': [3.0, 4.0],
            'R5Meta.2': [0.0, 0.0],
            'L5Meta': [5.0, 6.0],
            'L5Meta.1': [7.0, 8.0],
            'L5Meta.2': [0.0, 0.0],
        }
    )
    trc_data.columns = ['time', 'R5Meta', 'R5Meta', 'R5Meta', 'L5Meta', 'L5Meta', 'L5Meta']

    aliased = append_trc_marker_aliases(trc_data, marker_aliases=SYNTHPOSE_MARKER_ALIASES)

    assert aliased.columns.tolist() == [
        'time',
        'R5Meta', 'R5Meta', 'R5Meta',
        'L5Meta', 'L5Meta', 'L5Meta',
        'RSmallToe', 'RSmallToe', 'RSmallToe',
        'LSmallToe', 'LSmallToe', 'LSmallToe',
    ]
    assert np.allclose(aliased.iloc[:, -6:-3].to_numpy(), aliased.iloc[:, 1:4].to_numpy())
    assert np.allclose(aliased.iloc[:, -3:].to_numpy(), aliased.iloc[:, 4:7].to_numpy())


def test_append_trc_marker_aliases_preserves_existing_small_toe_triplets():
    '''
    Verify alias export does not duplicate markers that already exist.
    '''

    from Sports2D.process import append_trc_marker_aliases
    from Sports2D.Utilities.synthpose_skeleton import SYNTHPOSE_MARKER_ALIASES

    trc_data = pd.DataFrame(
        {
            'time': [0.0],
            'R5Meta': [1.0],
            'R5Meta.1': [2.0],
            'R5Meta.2': [0.0],
            'RSmallToe': [3.0],
            'RSmallToe.1': [4.0],
            'RSmallToe.2': [0.0],
        }
    )
    trc_data.columns = ['time', 'R5Meta', 'R5Meta', 'R5Meta', 'RSmallToe', 'RSmallToe', 'RSmallToe']

    aliased = append_trc_marker_aliases(trc_data, marker_aliases=SYNTHPOSE_MARKER_ALIASES)

    assert aliased.columns.tolist() == trc_data.columns.tolist()
    assert aliased.equals(trc_data)


def test_build_public_meter_trc_data_preserves_full_length_and_time_axis():
    '''
    Verify final public meter TRCs keep the original sample count and time axis.
    '''

    from Sports2D.process import build_public_meter_trc_data
    from Sports2D.Utilities.synthpose_skeleton import SYNTHPOSE_MARKER_ALIASES

    trc_data = pd.DataFrame(
        {
            'time': [0.10, 0.20, 0.30],
            'R5Meta': [1.0, 2.0, 3.0],
            'R5Meta.1': [4.0, 5.0, 6.0],
            'R5Meta.2': [0.0, 0.0, 0.0],
        },
        index=[10, 11, 12],
    )
    trc_data.columns = ['time', 'R5Meta', 'R5Meta', 'R5Meta']

    ball_trc_data = pd.DataFrame(
        {
            'ball': [10.0, np.nan, 30.0],
            'ball.1': [20.0, np.nan, 40.0],
            'ball.2': [0.0, np.nan, 0.0],
        },
        index=trc_data.index,
    )
    ball_trc_data.columns = ['ball', 'ball', 'ball']

    public_trc = build_public_meter_trc_data(
        trc_data,
        marker_aliases=SYNTHPOSE_MARKER_ALIASES,
        ball_trc_data=ball_trc_data,
        marker_name='ball',
    )

    assert public_trc.index.tolist() == trc_data.index.tolist()
    assert public_trc.iloc[:, 0].tolist() == pytest.approx(trc_data.iloc[:, 0].tolist())
    assert len(public_trc) == len(trc_data)
    assert public_trc.columns.tolist() == [
        'time',
        'R5Meta', 'R5Meta', 'R5Meta',
        'RSmallToe', 'RSmallToe', 'RSmallToe',
        'ball', 'ball', 'ball',
    ]
    assert np.isnan(public_trc.iloc[1, -3])
    assert np.isnan(public_trc.iloc[1, -2])
    assert np.isnan(public_trc.iloc[1, -1])


def test_reset_trc_frame_time_origin_rebases_trimmed_meter_exports():
    '''
    Verify trimmed TRC exports are rebased to a local frame/time origin.
    '''

    from Sports2D.process import reset_trc_frame_time_origin

    trc_data = pd.DataFrame(
        {
            'time': [0.4844026548672566, 0.5011061946902654],
            'Hip': [1.0, 2.0],
            'Hip.1': [3.0, 4.0],
            'Hip.2': [0.0, 0.0],
        },
        index=[29, 30],
    )
    trc_data.columns = ['time', 'Hip', 'Hip', 'Hip']

    rebased = reset_trc_frame_time_origin(trc_data)

    assert rebased.index.tolist() == [0, 1]
    assert rebased.iloc[:, 0].tolist() == pytest.approx([0.0, 0.01670353982300885])
    assert rebased.iloc[:, 1:].equals(trc_data.iloc[:, 1:].reset_index(drop=True))


def test_strip_auxiliary_trc_markers_removes_ball_before_reload():
    '''
    Verify TRC reload can ignore trailing ball markers added by save_pose exports.
    '''

    from Sports2D.process import strip_auxiliary_trc_markers

    q_coords = pd.DataFrame(
        np.array([
            [1.0, 2.0, 0.0, 10.0, 20.0, 0.0],
            [3.0, 4.0, 0.0, np.nan, np.nan, np.nan],
        ]),
        columns=['Hip', 'Hip', 'Hip', 'ball', 'ball', 'ball'],
    )

    filtered_q_coords, filtered_names = strip_auxiliary_trc_markers(
        q_coords,
        ['Hip', 'ball'],
        ignored_marker_names=('ball',),
    )

    assert filtered_names == ['Hip']
    assert filtered_q_coords.columns.tolist() == ['Hip', 'Hip', 'Hip']
    assert filtered_q_coords.shape == (2, 3)


def test_estimate_pelvis_trunk_com_y_uses_weighted_hip_neck_proxy():
    '''
    Verify the pelvis-trunk proxy CoM stays between Hip and Neck using the default alpha.
    '''

    from Sports2D.Utilities.motion import estimate_pelvis_trunk_com_y

    trc_data = pd.DataFrame(
        {
            'time': [0.0, 1.0 / 30.0],
            'Hip': [0.0, 0.0],
            'Hip.1': [1.0, 1.2],
            'Hip.2': [0.0, 0.0],
            'Neck': [0.0, 0.0],
            'Neck.1': [2.0, 2.2],
            'Neck.2': [0.0, 0.0],
        }
    )
    trc_data.columns = ['time', 'Hip', 'Hip', 'Hip', 'Neck', 'Neck', 'Neck']

    com_y = estimate_pelvis_trunk_com_y(trc_data)

    assert com_y.tolist() == pytest.approx([1.2, 1.4])


def test_estimate_pelvis_trunk_com_xy_px_returns_weighted_point():
    '''
    Verify pixel-space CoM overlay uses the same weighted Hip-Neck proxy.
    '''

    from Sports2D.Utilities.motion import estimate_pelvis_trunk_com_xy_px

    com_point = estimate_pelvis_trunk_com_xy_px(
        np.array([10.0, 10.0]),
        np.array([100.0, 50.0]),
        ['Hip', 'Neck'],
    )

    assert com_point == (10, 90)


def test_zero_flight_phase_sets_force_to_zero_between_takeoff_and_landing():
    '''
    Verify flight-phase cleanup removes force samples between take-off and landing.
    '''

    from Sports2D.Utilities.motion import _zero_flight_phase

    vgrf_n = np.array([700.0, 720.0, 25.0, 15.0, 705.0], dtype=float)

    zeroed = _zero_flight_phase(vgrf_n, takeoff_frame=2, landing_frame=4)

    assert zeroed.tolist() == pytest.approx([700.0, 720.0, 0.0, 0.0, 705.0])


def test_apply_vgrf_constraints_clamps_negative_values_to_zero():
    '''
    Verify exported vGRF is non-negative after flight cleanup and clamping.
    '''

    from Sports2D.Utilities.motion import _apply_vgrf_constraints

    constrained = _apply_vgrf_constraints(
        np.array([700.0, -25.0, -5.0, 20.0, 705.0], dtype=float),
        takeoff_frame=2,
        landing_frame=4,
    )

    assert constrained.tolist() == pytest.approx([700.0, 0.0, 0.0, 0.0, 705.0])


def test_analyze_vertical_jump_trial_returns_bodyweight_for_static_com():
    '''
    Verify a static CoM trajectory produces body-weight-only GRF.
    '''

    from Sports2D.Utilities.motion import analyze_vertical_jump_trial

    trc_data = pd.DataFrame(
        {
            'time': [0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0],
            'Hip': [0.0, 0.0, 0.0, 0.0],
            'Hip.1': [1.0, 1.0, 1.0, 1.0],
            'Hip.2': [0.0, 0.0, 0.0, 0.0],
            'Neck': [0.0, 0.0, 0.0, 0.0],
            'Neck.1': [1.5, 1.5, 1.5, 1.5],
            'Neck.2': [0.0, 0.0, 0.0, 0.0],
            'LBigToe': [0.0, 0.0, 0.0, 0.0],
            'LBigToe.1': [0.0, 0.0, 0.0, 0.0],
            'LBigToe.2': [0.0, 0.0, 0.0, 0.0],
            'RBigToe': [0.0, 0.0, 0.0, 0.0],
            'RBigToe.1': [0.0, 0.0, 0.0, 0.0],
            'RBigToe.2': [0.0, 0.0, 0.0, 0.0],
        }
    )
    trc_data.columns = [
        'time',
        'Hip', 'Hip', 'Hip',
        'Neck', 'Neck', 'Neck',
        'LBigToe', 'LBigToe', 'LBigToe',
        'RBigToe', 'RBigToe', 'RBigToe',
    ]

    result = analyze_vertical_jump_trial(trc_data, mass_kg=70.0, fps=30.0)

    assert result['body_weight_n'] == pytest.approx(70.0 * 9.81)
    assert result['vgrf_n'].tolist() == pytest.approx([70.0 * 9.81] * 4)
    assert result['metrics']['peak_vgrf_bw'] == pytest.approx(1.0)


def test_estimate_grf_arrow_anchor_px_projects_to_floor_line():
    '''
    Verify GRF arrows anchor at the support midpoint and floor line.
    '''

    from Sports2D.Utilities.motion import estimate_grf_arrow_anchor_px

    anchor = estimate_grf_arrow_anchor_px(
        np.array([100.0, 80.0, 140.0, 120.0, 30.0], dtype=float),
        np.array([510.0, 514.0, 508.0, 512.0, 60.0], dtype=float),
        ['LBigToe', 'LHeel', 'RBigToe', 'RHeel', 'Hip'],
        floor_x_origin=0.0,
        floor_y_origin=520.0,
        floor_angle=0.0,
    )

    assert anchor == (110, 520)


def test_resolve_vgrf_arrow_base_length_px_scales_with_frame_height():
    '''
    Verify the default GRF arrow base length scales with the video height.
    '''

    from Sports2D.Utilities.motion import resolve_vgrf_arrow_base_length_px

    assert resolve_vgrf_arrow_base_length_px(720) == 120
    assert resolve_vgrf_arrow_base_length_px(1080) == 180
    assert resolve_vgrf_arrow_base_length_px(2160) == 360


def test_project_force_to_arrow_length_px_uses_scaled_bodyweight_length():
    '''
    Verify the visual arrow scale maps force to the provided body-weight base length.
    '''

    from Sports2D.Utilities.motion import project_force_to_arrow_length_px

    assert project_force_to_arrow_length_px(700.0, 700.0, base_length_px=360.0) == 360
    assert project_force_to_arrow_length_px(1400.0, 700.0, base_length_px=360.0) == 720


def test_write_grf_trc_uses_newton_units(tmp_path):
    '''
    Verify GRF TRC export writes Newton units for the synthetic force marker.
    '''

    from Sports2D.Utilities.motion import write_grf_trc

    grf_path = tmp_path / 'GRF.trc'
    write_grf_trc(
        np.array([0.0, 1.0 / 30.0], dtype=float),
        np.array([700.0, 710.0], dtype=float),
        grf_path,
        fps=30.0,
    )

    contents = grf_path.read_text(encoding='utf-8')

    assert '\tN\t' in contents
    assert 'Frame#\tTime\tGRF' in contents


def test_lowpass_signal_requires_scipy_backend(monkeypatch):
    '''
    Verify vertical-jump filtering fails loudly when scipy's filter backend is unavailable.
    '''

    from Sports2D.Utilities import motion

    monkeypatch.setattr(motion, 'butter', None)
    monkeypatch.setattr(motion, 'filtfilt', None)

    with pytest.raises(RuntimeError, match='requires scipy.signal'):
        motion.lowpass_signal(np.array([1.0, 1.0, 1.0], dtype=float), fps=30.0)


def test_detect_vertical_jump_events_fallback_waits_for_propulsive_phase():
    '''
    Verify fallback take-off detection does not select frame 0 before any plausible propulsion.
    '''

    from Sports2D.Utilities.motion import detect_vertical_jump_events

    trc_data = pd.DataFrame({'time': [0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0, 4.0 / 30.0]})
    takeoff_frame, landing_frame = detect_vertical_jump_events(
        trc_data,
        com_velocity_y=np.array([0.0, 0.1, 0.35, 0.2, 0.0], dtype=float),
        raw_vgrf_n=np.array([10.0, 80.0, 700.0, 20.0, 710.0], dtype=float),
        body_weight_n=700.0,
        fps=30.0,
    )

    assert takeoff_frame == 3
    assert landing_frame == 4


def test_synthpose_tracker_merge_secondary_ball_detections_updates_ball_contract():
    '''
    Verify hybrid SAM3 ball detections merge into the existing tracker metadata schema.
    '''

    from Sports2D.Utilities.synthpose_tracker import (
        PERSON_CLASS_ID,
        SPORTS_BALL_CLASS_ID,
        SynthPosePoseTracker,
    )

    tracker = SynthPosePoseTracker.__new__(SynthPosePoseTracker)
    tracker.last_detections = {
        'boxes': np.array([[10, 20, 30, 40]], dtype=np.float32),
        'classes': np.array([PERSON_CLASS_ID], dtype=np.int32),
        'scores': np.array([0.95], dtype=np.float32),
        'person_boxes': np.array([[10, 20, 30, 40]], dtype=np.float32),
        'ball_boxes': np.empty((0, 4), dtype=np.float32),
        'ball_scores': np.empty((0,), dtype=np.float32),
        'class_names': np.array(['person'], dtype=object),
        'prompt_indices': np.array([-1], dtype=np.int32),
        'sam3_ball_meta': {},
    }

    tracker._merge_secondary_ball_detections({
        'boxes': np.array([[50, 60, 70, 80]], dtype=np.float32),
        'classes': np.array([SPORTS_BALL_CLASS_ID], dtype=np.int32),
        'scores': np.array([0.81], dtype=np.float32),
        'person_boxes': np.empty((0, 4), dtype=np.float32),
        'ball_boxes': np.array([[50, 60, 70, 80]], dtype=np.float32),
        'ball_scores': np.array([0.81], dtype=np.float32),
        'class_names': np.array(['sports ball'], dtype=object),
        'prompt_indices': np.array([0], dtype=np.int32),
        'masks': [np.ones((2, 2), dtype=np.uint8)],
    })

    merged = tracker.last_detections
    assert merged['classes'].tolist() == [PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID]
    assert merged['ball_boxes'].shape == (1, 4)
    assert merged['ball_scores'].tolist() == pytest.approx([0.81])
    assert merged['sam3_ball_meta']['class_names'].tolist() == ['sports ball']


def test_synthpose_tracker_secondary_sam3_ball_detection_runs_on_skipped_frames():
    '''
    Verify the dedicated SAM3 ball detector still runs on frames where person detection is skipped by cadence.
    '''

    from Sports2D.Utilities.synthpose_tracker import (
        PERSON_CLASS_ID,
        SPORTS_BALL_CLASS_ID,
        SynthPosePoseTracker,
    )

    tracker = SynthPosePoseTracker.__new__(SynthPosePoseTracker)
    tracker.frame_count = 0
    tracker.det_frequency = 3
    tracker.prev_boxes = None
    tracker.detect_ball = True
    tracker.ball_detector_backend = 'sam3'
    tracker.detector_type = 'yolox'
    tracker.sam3_collect_masks = False
    tracker.last_detections = tracker._empty_detections()

    person_detection_calls = []
    ball_detection_calls = []

    def fake_detect_persons(frame, height=None, width=None):
        person_detection_calls.append((height, width))
        return np.array([[8.0, 10.0, 12.0, 16.0]], dtype=np.float32)

    def fake_detect_balls(frame):
        ball_detection_calls.append(frame.shape[:2])
        return {
            'boxes': np.array([[50.0, 60.0, 62.0, 72.0]], dtype=np.float32),
            'classes': np.array([SPORTS_BALL_CLASS_ID], dtype=np.int32),
            'scores': np.array([0.81], dtype=np.float32),
            'person_boxes': np.empty((0, 4), dtype=np.float32),
            'ball_boxes': np.array([[50.0, 60.0, 62.0, 72.0]], dtype=np.float32),
            'ball_scores': np.array([0.81], dtype=np.float32),
            'class_names': np.array(['sports ball'], dtype=object),
            'prompt_indices': np.array([0], dtype=np.int32),
            'sam3_ball_meta': {},
        }

    tracker._detect_persons = fake_detect_persons
    tracker._detect_balls_sam3 = fake_detect_balls
    tracker._estimate_poses = lambda pil_image, person_boxes: (
        np.zeros((1, 52, 2), dtype=np.float32),
        np.zeros((1, 52), dtype=np.float32),
    )

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    tracker(frame)
    tracker(frame)

    assert len(person_detection_calls) == 1
    assert len(ball_detection_calls) == 2
    assert tracker.last_detections['ball_boxes'].shape == (1, 4)
    assert tracker.last_detections['classes'].tolist() == [SPORTS_BALL_CLASS_ID]


def test_synthpose_tracker_yolo26_adapter_builds_detection_metadata():
    '''
    Verify the YOLO26 adapter preserves low-threshold metadata while filtering person boxes for VitPose.
    '''

    from Sports2D.Utilities.synthpose_tracker import (
        SynthPosePoseTracker,
        SPORTS_BALL_CLASS_ID,
    )

    class FakeBoxes:
        def __init__(self):
            self.xyxy = np.array([
                [10.0, 20.0, 30.0, 40.0],
                [50.0, 60.0, 70.0, 84.0],
                [5.0, 8.0, 16.0, 24.0],
            ], dtype=np.float32)
            self.cls = np.array([0, SPORTS_BALL_CLASS_ID, 0], dtype=np.float32)
            self.conf = np.array([0.81, 0.72, 0.31], dtype=np.float32)

    class FakeResult:
        def __init__(self):
            self.boxes = FakeBoxes()
            self.names = {
                0: 'person',
                SPORTS_BALL_CLASS_ID: 'sports ball',
            }

    class FakeDetector:
        def __call__(self, frame, **kwargs):
            assert kwargs.get('verbose') is False
            return [FakeResult()]

    tracker = SynthPosePoseTracker.__new__(SynthPosePoseTracker)
    tracker.detector = FakeDetector()
    tracker.person_threshold = 0.5
    tracker.ball_detection_threshold = 0.2
    tracker.ball_class_ids = [SPORTS_BALL_CLASS_ID]
    tracker.detect_ball = True
    tracker.ball_detector_backend = 'same'
    tracker.detector_type = 'yolo26'
    tracker.sam3_collect_masks = False
    tracker.last_detections = tracker._empty_detections()

    person_boxes = tracker._detect_persons_yolo26(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    metadata = tracker.last_detections

    assert person_boxes.shape == (1, 4)
    assert person_boxes[0].tolist() == pytest.approx([10.0, 20.0, 20.0, 20.0])
    assert metadata['boxes'].shape == (3, 4)
    assert metadata['person_boxes'].shape == (1, 4)
    assert metadata['ball_boxes'].shape == (1, 4)
    assert metadata['ball_scores'].tolist() == pytest.approx([0.72])
    assert metadata['class_names'].tolist() == ['person', 'sports ball', 'person']


def test_resolve_sam3_runtime_auto_switches_raw_checkpoint_to_meta(tmp_path):
    '''
    Verify raw SAM3 checkpoints bypass processor_path and use the Meta runtime.
    '''

    from Sports2D.Utilities.sam3_detector import (
        is_sam3_checkpoint_path,
        resolve_sam3_runtime,
    )

    checkpoint_path = tmp_path / 'sam3.pt'
    checkpoint_path.write_bytes(b'')

    assert is_sam3_checkpoint_path(str(checkpoint_path))
    assert resolve_sam3_runtime('transformers', str(checkpoint_path)) == 'meta'
    assert resolve_sam3_runtime('auto', str(checkpoint_path)) == 'meta'
    assert resolve_sam3_runtime('official', str(checkpoint_path)) == 'meta'


def test_resolve_sam3_runtime_keeps_transformers_for_hf_bundle(tmp_path):
    '''
    Verify non-checkpoint SAM3 paths stay on the transformers runtime by default.
    '''

    from Sports2D.Utilities.sam3_detector import (
        is_sam3_checkpoint_path,
        resolve_sam3_runtime,
    )

    bundle_dir = tmp_path / 'sam3_bundle'
    bundle_dir.mkdir()
    (bundle_dir / 'config.json').write_text('{}', encoding='utf-8')

    assert not is_sam3_checkpoint_path(str(bundle_dir))
    assert resolve_sam3_runtime('transformers', str(bundle_dir)) == 'transformers'
    assert resolve_sam3_runtime('auto', str(bundle_dir)) == 'transformers'


def test_default_config_exposes_realtime_ui_options():
    '''
    Verify realtime UI options are present in config defaults and CLI help.
    '''

    from Sports2D.Sports2D import DEFAULT_CONFIG, CONFIG_HELP

    assert DEFAULT_CONFIG['base']['realtime_ui_backend'] == 'opencv'
    assert DEFAULT_CONFIG['base']['realtime_window_title'] == 'UmFit realtime'
    assert 'realtime_ui_backend' in CONFIG_HELP
    assert 'realtime_window_title' in CONFIG_HELP


def test_default_config_exposes_video_codec():
    '''
    Verify output video codec is configurable from defaults and CLI help.
    '''

    from Sports2D.Sports2D import DEFAULT_CONFIG, CONFIG_HELP

    assert DEFAULT_CONFIG['base']['video_codec'] == 'mp4v'
    assert 'video_codec' in CONFIG_HELP


def test_default_config_exposes_ball_ordering_method():
    '''
    Verify ball ordering config is exposed in defaults and help text.
    '''

    from Sports2D.Sports2D import DEFAULT_CONFIG, CONFIG_HELP

    assert DEFAULT_CONFIG['pose']['ball_ordering_method'] == 'first_detected'
    assert 'ball_ordering_method' in CONFIG_HELP


def test_config_help_mentions_ball_pose_exports():
    '''
    Verify help text explains the new ball export behavior.
    '''

    from Sports2D.Sports2D import CONFIG_HELP

    assert 'pose_ball/' in CONFIG_HELP['save_pose'][1]
    assert 'pose_ball/' in CONFIG_HELP['detect_ball'][1]
    assert 'Blender helper script' in CONFIG_HELP['save_pose'][1]
    assert 'Blender helper script' in CONFIG_HELP['detect_ball'][1]


def test_default_config_exposes_sam3_settings():
    '''
    Verify SAM3 config surface is available from defaults and CLI help.
    '''

    from Sports2D.Sports2D import DEFAULT_CONFIG, CONFIG_HELP

    assert DEFAULT_CONFIG['pose']['sam3_target'] == 'ball'
    assert DEFAULT_CONFIG['pose']['sam3_runtime'] == 'transformers'
    assert DEFAULT_CONFIG['pose']['ball_detector_backend'] == 'same'
    assert DEFAULT_CONFIG['pose']['sam3_show_realtime_masks'] is False
    assert DEFAULT_CONFIG['pose']['sam3_realtime_mask_alpha'] == pytest.approx(0.22)
    assert 'ball_detector_backend' in CONFIG_HELP
    assert 'sam3_target' in CONFIG_HELP
    assert 'sam3_model_path' in CONFIG_HELP
    assert 'sam3_processor_path' in CONFIG_HELP
    assert 'sam3_show_realtime_masks' in CONFIG_HELP
    assert 'sam3_realtime_mask_alpha' in CONFIG_HELP


def test_config_help_mentions_yolo26_detector():
    '''
    Verify SynthPose detector help text lists YOLO26.
    '''

    from Sports2D.Sports2D import CONFIG_HELP

    assert 'yolo26' in CONFIG_HELP['synthpose_detector'][1]


def test_default_config_exposes_vertical_jump_motion_mode():
    '''
    Verify vertical jump motion mode is available in defaults and CLI help.
    '''

    from Sports2D.Sports2D import DEFAULT_CONFIG, CONFIG_HELP

    assert DEFAULT_CONFIG['motion']['vertical_jump'] is False
    assert 'vertical_jump' in CONFIG_HELP


def test_default_config_exposes_hybrid_review_settings():
    '''
    Verify hybrid manual-review settings are exposed in defaults and CLI help.
    '''

    from Sports2D.Sports2D import DEFAULT_CONFIG, CONFIG_HELP

    assert DEFAULT_CONFIG['base']['hybrid_mode'] is False
    assert DEFAULT_CONFIG['base']['hybrid_review_pose'] is True
    assert DEFAULT_CONFIG['base']['hybrid_review_ball'] is True
    assert DEFAULT_CONFIG['base']['hybrid_ui_backend'] == 'matplotlib'
    assert 'hybrid_mode' in CONFIG_HELP
    assert 'hybrid_review_pose' in CONFIG_HELP
    assert 'hybrid_review_ball' in CONFIG_HELP
    assert 'hybrid_ui_backend' in CONFIG_HELP


def test_normalize_hybrid_ui_backend_accepts_supported_values():
    '''
    Verify hybrid UI backend normalization preserves supported values and defaults unknown input.
    '''

    from Sports2D.Utilities.hybrid_editor import normalize_hybrid_ui_backend

    assert normalize_hybrid_ui_backend(None) == 'auto'
    assert normalize_hybrid_ui_backend('qt') == 'qt'
    assert normalize_hybrid_ui_backend(' matplotlib ') == 'matplotlib'
    assert normalize_hybrid_ui_backend('bogus') == 'auto'


def test_review_pose_sequence_dispatches_to_qt_backend(monkeypatch):
    '''
    Verify pose review routes through the Qt backend when requested and available.
    '''

    from Sports2D.Utilities import hybrid_editor

    expected = ('x', 'y', 'scores', 'mask')

    class DummyQtModule:
        @staticmethod
        def review_pose_sequence_qt(**kwargs):
            return kwargs['person_x_raw'], kwargs['person_y_raw'], kwargs['person_scores_raw'], kwargs['manual_mask']

    monkeypatch.setattr(hybrid_editor, '_load_qt_hybrid_editor_module', lambda: DummyQtModule)

    result = hybrid_editor.review_pose_sequence(
        video_file_path='demo.mp4',
        frame_range=(0, 1),
        person_x_raw=expected[0],
        person_y_raw=expected[1],
        person_scores_raw=expected[2],
        keypoint_names=['Nose'],
        keypoint_threshold=0.3,
        manual_mask=expected[3],
        ui_backend='qt',
    )

    assert result == expected


def test_review_ball_sequence_falls_back_to_matplotlib_backend_when_qt_fails(monkeypatch):
    '''
    Verify ball review falls back to the Matplotlib backend when Qt import/init fails.
    '''

    from Sports2D.Utilities import hybrid_editor

    expected = (['center'], ['visible'], ['mask'])

    def raise_qt_error():
        raise ImportError('PySide6 missing')

    def fake_matplotlib_backend(**kwargs):
        return kwargs['ball_centers'], kwargs['ball_boxes'], kwargs['ball_scores']

    monkeypatch.setattr(hybrid_editor, '_load_qt_hybrid_editor_module', raise_qt_error)
    monkeypatch.setattr(hybrid_editor, '_review_ball_sequence_matplotlib', fake_matplotlib_backend)

    result = hybrid_editor.review_ball_sequence(
        video_file_path='demo.mp4',
        frame_range=(0, 1),
        ball_centers=expected[0],
        ball_boxes=expected[1],
        ball_scores=expected[2],
        ball_tracks=[],
        selected_ball_ids=[],
        ui_backend='qt',
    )

    assert result == expected


def test_selected_track_review_state_recovers_visible_source_track_for_stitched_id():
    '''
    Verify hybrid review can resolve score/visibility from the visible raw fragment for a stitched selected ID.
    '''

    from Sports2D.Utilities.hybrid_editor import _selected_track_review_state

    center, score, visible, source_track_id = _selected_track_review_state(
        [
            {
                'id': 19,
                'center': (30, 20),
                'box': np.array([24.0, 14.0, 36.0, 26.0], dtype=np.float32),
                'score': 0.81,
                'visible': True,
            },
        ],
        selected_track_id=7,
        frame_center=(30, 20),
    )

    assert center == (30, 20)
    assert score == pytest.approx(0.81)
    assert visible is True
    assert source_track_id == 19


def test_review_pose_sequence_qt_returns_original_values_when_dialog_rejected(monkeypatch):
    '''
    Verify Qt pose review preserves the original arrays when the dialog is canceled.
    '''

    from Sports2D.Utilities import hybrid_editor_qt

    class DummyQApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, _args):
            pass

        def processEvents(self, *_args):
            return None

    class DummyPoseDialog:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

        def exec(self):
            return 0

        def get_result(self):
            return ('changed_x', 'changed_y', 'changed_scores', 'changed_mask')

    class DummyDialogCode:
        Accepted = 1

    class DummyQDialog:
        DialogCode = DummyDialogCode

    class DummyProcessEventsFlag:
        AllEvents = object()

    class DummyEventLoop:
        ProcessEventsFlag = DummyProcessEventsFlag

    monkeypatch.setattr(hybrid_editor_qt, '_require_qt', lambda: None)
    monkeypatch.setattr(hybrid_editor_qt, 'QApplication', DummyQApplication)
    monkeypatch.setattr(hybrid_editor_qt, 'PoseReviewDialog', DummyPoseDialog)
    monkeypatch.setattr(hybrid_editor_qt, 'QDialog', DummyQDialog)
    monkeypatch.setattr(hybrid_editor_qt, 'QEventLoop', DummyEventLoop)

    expected_x = np.array([[10.0, 20.0]], dtype=float)
    expected_y = np.array([[30.0, 40.0]], dtype=float)
    expected_scores = np.array([[0.9, 0.8]], dtype=float)
    expected_mask = np.array([[True, False]], dtype=bool)

    result = hybrid_editor_qt.review_pose_sequence_qt(
        video_file_path='demo.mp4',
        frame_range=(0, 1),
        person_x_raw=expected_x,
        person_y_raw=expected_y,
        person_scores_raw=expected_scores,
        keypoint_names=['LHip', 'RHip'],
        keypoint_threshold=0.3,
        manual_mask=expected_mask,
    )

    assert np.array_equal(result[0], expected_x)
    assert np.array_equal(result[1], expected_y)
    assert np.array_equal(result[2], expected_scores)
    assert np.array_equal(result[3], expected_mask)


def test_review_ball_sequence_qt_returns_original_values_when_dialog_rejected(monkeypatch):
    '''
    Verify Qt ball review preserves the original timeline when the dialog is canceled.
    '''

    from Sports2D.Utilities import hybrid_editor_qt

    class DummyQApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, _args):
            pass

        def processEvents(self, *_args):
            return None

    class DummyBallDialog:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

        def exec(self):
            return 0

        def get_result(self):
            return (['changed_center'], ['changed_visible'], ['changed_mask'])

    class DummyDialogCode:
        Accepted = 1

    class DummyQDialog:
        DialogCode = DummyDialogCode

    class DummyProcessEventsFlag:
        AllEvents = object()

    class DummyEventLoop:
        ProcessEventsFlag = DummyProcessEventsFlag

    monkeypatch.setattr(hybrid_editor_qt, '_require_qt', lambda: None)
    monkeypatch.setattr(hybrid_editor_qt, 'QApplication', DummyQApplication)
    monkeypatch.setattr(hybrid_editor_qt, 'BallReviewDialog', DummyBallDialog)
    monkeypatch.setattr(hybrid_editor_qt, 'QDialog', DummyQDialog)
    monkeypatch.setattr(hybrid_editor_qt, 'QEventLoop', DummyEventLoop)

    result = hybrid_editor_qt.review_ball_sequence_qt(
        video_file_path='demo.mp4',
        frame_range=(0, 1),
        ball_centers=[(50, 60), None],
        ball_boxes=[np.empty((0, 4), dtype=float), np.empty((0, 4), dtype=float)],
        ball_scores=[np.empty((0,), dtype=float), np.empty((0,), dtype=float)],
        ball_tracks=[[], []],
        selected_ball_ids=[7, 7],
    )

    assert result == ([(50, 60), None], [True, False], [False, False])


def test_keypoint_names_in_output_order_matches_synthpose_tensor_order():
    '''
    Verify output-order keypoint names match the actual SynthPose tensor index order.
    '''

    from Sports2D.Utilities.pose_backend import _keypoint_names_in_output_order
    from Sports2D.Utilities.synthpose_skeleton import (
        SYNTHPOSE_KEYPOINT_NAMES,
        create_synthpose_skeleton,
    )

    assert _keypoint_names_in_output_order(create_synthpose_skeleton()) == SYNTHPOSE_KEYPOINT_NAMES


def test_rtmlib_ball_tracker_honors_detection_frequency_cadence():
    '''
    Verify the RTMLib ball detector only runs on cadence frames.
    '''

    from Sports2D.Utilities.pose_backend import (
        _RTMLibBallAwareTracker,
        SPORTS_BALL_CLASS_ID,
    )

    tracker = _RTMLibBallAwareTracker.__new__(_RTMLibBallAwareTracker)
    tracker._pose_tracker = lambda frame: (
        np.array([[[10.0, 10.0], [14.0, 16.0]]], dtype=np.float32),
        np.array([[0.9, 0.8]], dtype=np.float32),
    )
    detector_calls = []

    def fake_ball_detector(frame):
        detector_calls.append(frame.shape)
        return (
            np.array([[20.0, 30.0, 40.0, 50.0]], dtype=np.float32),
            np.array([SPORTS_BALL_CLASS_ID], dtype=np.int32),
            np.array([0.93], dtype=np.float32),
        )

    tracker._ball_detector = fake_ball_detector
    tracker._ball_class_ids = {SPORTS_BALL_CLASS_ID}
    tracker._det_frequency = 2
    tracker._frame_count = 0
    tracker.last_detections = tracker._empty_detections()

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    tracker(frame)
    tracker(frame)
    tracker(frame)

    assert len(detector_calls) == 2
    assert tracker.last_detections['ball_boxes'].shape == (1, 4)


class _FakeHybridCapture:
    def __init__(self, frame_count):
        self.frames = [
            np.full((4, 4, 3), fill_value=frame_idx, dtype=np.uint8)
            for frame_idx in range(frame_count)
        ]
        self.position = 0
        self.set_calls = []
        self.read_calls = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, _prop_id, value):
        self.position = int(value)
        self.set_calls.append(int(value))
        return True

    def read(self):
        self.read_calls += 1
        if self.position < 0 or self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def test_video_frame_navigator_uses_sequential_reads_for_adjacent_frames():
    '''
    Verify adjacent forward frames avoid an extra seek after the first decode.
    '''

    from Sports2D.Utilities.hybrid_editor import VideoFrameNavigator

    capture = _FakeHybridCapture(frame_count=12)
    navigator = VideoFrameNavigator(capture, start_frame=0, cache_size=8, sequential_window=4)

    frame0 = navigator.get_frame(0)
    frame1 = navigator.get_frame(1)

    assert int(frame0[0, 0, 0]) == 0
    assert int(frame1[0, 0, 0]) == 1
    assert capture.set_calls == [0]
    assert capture.read_calls == 2


def test_video_frame_navigator_reuses_cached_previous_frame_without_seek():
    '''
    Verify returning to a recently viewed frame reuses cache instead of seeking again.
    '''

    from Sports2D.Utilities.hybrid_editor import VideoFrameNavigator

    capture = _FakeHybridCapture(frame_count=12)
    navigator = VideoFrameNavigator(capture, start_frame=0, cache_size=8, sequential_window=4)

    navigator.get_frame(0)
    navigator.get_frame(1)
    frame0_again = navigator.get_frame(0)

    assert int(frame0_again[0, 0, 0]) == 0
    assert capture.set_calls == [0]
    assert capture.read_calls == 2


def test_video_frame_navigator_seeks_once_for_far_jump():
    '''
    Verify large jumps still use a single direct seek/read path.
    '''

    from Sports2D.Utilities.hybrid_editor import VideoFrameNavigator

    capture = _FakeHybridCapture(frame_count=20)
    navigator = VideoFrameNavigator(capture, start_frame=0, cache_size=8, sequential_window=4)

    navigator.get_frame(0)
    frame10 = navigator.get_frame(10)

    assert int(frame10[0, 0, 0]) == 10
    assert capture.set_calls == [0, 10]
    assert capture.read_calls == 2


def test_frame_render_controller_coalesces_to_latest_pending_frame():
    '''
    Verify multiple frame requests collapse to the latest target before rendering.
    '''

    from Sports2D.Utilities.hybrid_editor import FrameRenderController

    controller = FrameRenderController()
    controller.request(1)
    controller.request(4)
    controller.request(7)

    assert controller.consume() == 7

    controller.request(7)
    assert controller.consume() is None

    controller.request(3)
    assert controller.consume() == 3


def test_compute_zoomed_limits_zooms_in_around_focus():
    '''
    Verify zoom-in keeps the cursor-side focus while shrinking the visible span.
    '''

    from Sports2D.Utilities.hybrid_editor import _compute_zoomed_limits

    zoomed = _compute_zoomed_limits(
        current_limits=(0.0, 100.0),
        focus_value=75.0,
        zoom_factor=0.5,
        bounds_limits=(0.0, 100.0),
        min_span=20.0,
    )

    assert np.allclose(zoomed, (37.5, 87.5))


def test_compute_zoomed_limits_preserves_inverted_axis_and_bounds():
    '''
    Verify zoom logic respects image-style inverted axes and clamps to frame bounds.
    '''

    from Sports2D.Utilities.hybrid_editor import _compute_zoomed_limits

    zoomed = _compute_zoomed_limits(
        current_limits=(100.0, 0.0),
        focus_value=100.0,
        zoom_factor=0.5,
        bounds_limits=(100.0, 0.0),
        min_span=20.0,
    )
    zoomed_out = _compute_zoomed_limits(
        current_limits=(75.0, 25.0),
        focus_value=50.0,
        zoom_factor=4.0,
        bounds_limits=(100.0, 0.0),
        min_span=20.0,
    )

    assert np.allclose(zoomed, (100.0, 50.0))
    assert np.allclose(zoomed_out, (100.0, 0.0))


def test_qt_clamp_view_box_keeps_view_inside_image_bounds():
    '''
    Verify Qt canvas view boxes are clamped to the available image area.
    '''

    from Sports2D.Utilities.hybrid_editor_qt import _clamp_view_box

    clamped = _clamp_view_box(
        left=-15.0,
        top=95.0,
        width=40.0,
        height=30.0,
        image_width=100.0,
        image_height=100.0,
    )

    assert clamped == (0.0, 70.0, 40.0, 30.0)


def test_qt_translate_view_box_moves_with_middle_drag_and_clamps():
    '''
    Verify middle-drag pan translates the Qt canvas view and respects image bounds.
    '''

    from Sports2D.Utilities.hybrid_editor_qt import _translate_view_box

    translated = _translate_view_box(
        left=10.0,
        top=20.0,
        width=50.0,
        height=40.0,
        widget_dx=40.0,
        widget_dy=-20.0,
        target_width=200.0,
        target_height=100.0,
        image_width=120.0,
        image_height=100.0,
    )
    translated_clamped = _translate_view_box(
        left=0.0,
        top=0.0,
        width=50.0,
        height=40.0,
        widget_dx=200.0,
        widget_dy=200.0,
        target_width=200.0,
        target_height=100.0,
        image_width=120.0,
        image_height=100.0,
    )

    assert np.allclose(translated, (0.0, 28.0, 50.0, 40.0))
    assert np.allclose(translated_clamped, (0.0, 0.0, 50.0, 40.0))


def test_evaluate_pose_frame_rejects_person_when_too_few_keypoints_remain():
    '''
    Verify frame-level pose filtering rejects a person when too few keypoints survive.
    '''

    from Sports2D.Utilities.hybrid_editor import evaluate_pose_frame

    filtered_x, filtered_y, filtered_scores, rejection_reason = evaluate_pose_frame(
        raw_x=np.array([10.0, 20.0, 30.0], dtype=float),
        raw_y=np.array([15.0, 25.0, 35.0], dtype=float),
        raw_scores=np.array([0.9, 0.2, 0.1], dtype=float),
        keypoint_threshold=0.3,
        average_threshold=0.5,
        keypoint_number_threshold=0.6,
    )

    assert rejection_reason == 'too_few_keypoints'
    assert np.isnan(filtered_x).all()
    assert np.isnan(filtered_y).all()
    assert np.isnan(filtered_scores).all()


def test_evaluate_pose_frame_keeps_keypoints_when_thresholds_pass():
    '''
    Verify frame-level pose filtering preserves coordinates when thresholds pass.
    '''

    from Sports2D.Utilities.hybrid_editor import evaluate_pose_frame

    filtered_x, filtered_y, filtered_scores, rejection_reason = evaluate_pose_frame(
        raw_x=np.array([10.0, 20.0, 30.0], dtype=float),
        raw_y=np.array([15.0, 25.0, 35.0], dtype=float),
        raw_scores=np.array([0.9, 0.4, 0.35], dtype=float),
        keypoint_threshold=0.3,
        average_threshold=0.5,
        keypoint_number_threshold=0.6,
    )

    assert rejection_reason is None
    assert filtered_x.tolist() == pytest.approx([10.0, 20.0, 30.0])
    assert filtered_y.tolist() == pytest.approx([15.0, 25.0, 35.0])
    assert filtered_scores.tolist() == pytest.approx([0.9, 0.4, 0.35])


def test_build_pose_issue_list_reports_missing_low_confidence_and_manual_points():
    '''
    Verify pose diagnostics surface missing, low-confidence, and manual statuses.
    '''

    from Sports2D.Utilities.hybrid_editor import build_pose_issue_list

    issues = build_pose_issue_list(
        frame_x=np.array([100.0, np.nan, 220.0], dtype=float),
        frame_y=np.array([120.0, np.nan, 260.0], dtype=float),
        frame_scores=np.array([0.95, 0.85, 0.22], dtype=float),
        keypoint_names=['Nose', 'RWrist', 'LWrist'],
        keypoint_threshold=0.3,
        manual_mask_frame=np.array([True, False, False], dtype=bool),
        frame_index=1,
        full_x_series=np.array([
            [99.0, 140.0, 210.0],
            [100.0, np.nan, 220.0],
            [101.0, 142.0, 222.0],
        ], dtype=float),
        full_y_series=np.array([
            [119.0, 160.0, 250.0],
            [120.0, np.nan, 260.0],
            [121.0, 162.0, 262.0],
        ], dtype=float),
    )

    statuses = [issue['status'] for issue in issues]
    assert statuses == ['missing', 'low_confidence', 'manually_edited']
    assert issues[0]['keypoint'] == 'RWrist'
    assert issues[0]['ghost_xy'] == pytest.approx((140.0, 160.0))
    assert issues[1]['keypoint'] == 'LWrist'
    assert issues[1]['score'] == pytest.approx(0.22)
    assert issues[2]['keypoint'] == 'Nose'
    assert issues[2]['manual'] is True


def test_build_pose_issue_list_reports_derived_points_as_read_only():
    '''
    Verify derived keypoints are surfaced separately and marked read-only.
    '''

    from Sports2D.Utilities.hybrid_editor import build_pose_issue_list

    issues = build_pose_issue_list(
        frame_x=np.array([100.0, 140.0], dtype=float),
        frame_y=np.array([120.0, 160.0], dtype=float),
        frame_scores=np.array([0.95, 0.9], dtype=float),
        keypoint_names=['Nose', 'Hip'],
        keypoint_threshold=0.3,
    )

    assert len(issues) == 1
    assert issues[0]['keypoint'] == 'Hip'
    assert issues[0]['status'] == 'derived'
    assert issues[0]['editable'] is False


def test_augment_pose_arrays_with_derived_keypoints_appends_review_markers():
    '''
    Verify hybrid review augments raw pose arrays with derived Hip/Neck markers.
    '''

    from Sports2D.Utilities.hybrid_editor import augment_pose_arrays_with_derived_keypoints

    augmented_x, augmented_y, augmented_scores, keypoint_names = augment_pose_arrays_with_derived_keypoints(
        person_x_raw=np.array([[10.0, 30.0, 50.0, 70.0]], dtype=float),
        person_y_raw=np.array([[100.0, 120.0, 140.0, 160.0]], dtype=float),
        person_scores_raw=np.array([[0.8, 0.6, 0.9, 0.7]], dtype=float),
        keypoint_names=['LHip', 'RHip', 'LShoulder', 'RShoulder'],
    )

    assert keypoint_names == ['LHip', 'RHip', 'LShoulder', 'RShoulder', 'Hip', 'Neck']
    assert augmented_x[0, 4] == pytest.approx(20.0)
    assert augmented_y[0, 4] == pytest.approx(110.0)
    assert augmented_scores[0, 4] == pytest.approx(0.7)
    assert augmented_x[0, 5] == pytest.approx(60.0)
    assert augmented_y[0, 5] == pytest.approx(150.0)
    assert augmented_scores[0, 5] == pytest.approx(0.8)


def test_build_ball_issue_list_reports_missing_low_confidence_and_manual_override():
    '''
    Verify ball diagnostics capture missing, low-confidence, and manual statuses.
    '''

    from Sports2D.Utilities.hybrid_editor import build_ball_issue_list

    issues = build_ball_issue_list(
        center=None,
        score=0.05,
        score_threshold=0.1,
        manual_override=True,
        visible=False,
        track_missing=True,
    )

    statuses = [issue['status'] for issue in issues]
    assert statuses == ['missing_ball', 'track_gap', 'low_confidence_ball', 'manual_ball_override']


def test_apply_ball_override_to_tracks_updates_selected_track_metadata():
    '''
    Verify manual ball overrides rewrite the selected track center and visibility.
    '''

    from Sports2D.Utilities.hybrid_editor import apply_ball_override_to_tracks

    tracks = [
        {
            'id': 7,
            'center': (20, 30),
            'box': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            'score': 0.88,
            'visible': True,
            'missing': 0,
        }
    ]

    updated_tracks = apply_ball_override_to_tracks(
        tracks,
        selected_track_id=7,
        center=(50, 60),
        visible=True,
    )

    assert updated_tracks[0]['center'] == (50, 60)
    assert updated_tracks[0]['visible'] is True
    assert updated_tracks[0]['missing'] == 0
    assert updated_tracks[0]['box'].tolist() == pytest.approx([40.0, 50.0, 60.0, 70.0])


def test_create_pose_backend_surfaces_raw_sam3_guidance(monkeypatch):
    '''
    Verify raw SAM3 checkpoint import failures surface runtime-specific guidance.
    '''

    from Sports2D.Utilities import pose_backend

    def fake_synthpose_backend(_config):
        raise ImportError(
            "Raw SAM3 checkpoints (.pt/.pth) require the official Meta sam3 package."
        )

    monkeypatch.setattr(pose_backend, 'SynthPoseBackend', fake_synthpose_backend)

    with pytest.raises(ImportError) as exc_info:
        pose_backend.create_pose_backend({'pose': {'pose_model': 'synthpose'}})

    message = str(exc_info.value)
    assert "official Meta sam3 package" in message
    assert "sam3_runtime='transformers'" in message
    assert "sports2d[synthpose]" not in message


def test_create_pose_backend_surfaces_ultralytics_guidance(monkeypatch):
    '''
    Verify missing Ultralytics dependency surfaces detector-specific install guidance.
    '''

    from Sports2D.Utilities import pose_backend

    def fake_synthpose_backend(_config):
        raise ImportError(
            "Ultralytics YOLO detectors require the 'ultralytics' package."
        )

    monkeypatch.setattr(pose_backend, 'SynthPoseBackend', fake_synthpose_backend)

    with pytest.raises(ImportError) as exc_info:
        pose_backend.create_pose_backend({
            'pose': {
                'pose_model': 'synthpose',
                'synthpose_detector': 'yolo26',
            }
        })

    message = str(exc_info.value)
    assert "sports2d[synthpose,yolo26]" in message
    assert "torch transformers ultralytics" in message


def test_create_pose_backend_surfaces_transformers_sam3_guidance(monkeypatch):
    '''
    Verify transformers SAM3 import failures surface version-upgrade guidance.
    '''

    from Sports2D.Utilities import pose_backend

    def fake_synthpose_backend(_config):
        raise ImportError(
            "Hugging Face SAM3 runtime requires a transformers build that exposes "
            "Sam3Model/Sam3Processor."
        )

    monkeypatch.setattr(pose_backend, 'SynthPoseBackend', fake_synthpose_backend)

    with pytest.raises(ImportError) as exc_info:
        pose_backend.create_pose_backend({'pose': {'pose_model': 'synthpose'}})

    message = str(exc_info.value)
    assert "current transformers install in this environment is too old for SAM3" in message
    assert "git+https://github.com/huggingface/transformers" in message
    assert "Raw SAM3 checkpoint mode is separate" not in message


def test_get_start_time_ffmpeg_uses_utf8_decoding(monkeypatch):
    '''
    Verify ffmpeg metadata probe decodes stderr safely and parses the start time.
    '''

    from Sports2D.Utilities import common

    captured = {}

    def fake_get_ffmpeg_exe():
        return '/tmp/fake_ffmpeg'

    def fake_run(cmd, stderr=None, stdout=None, text=None, encoding=None, errors=None):
        captured['cmd'] = cmd
        captured['stderr'] = stderr
        captured['stdout'] = stdout
        captured['text'] = text
        captured['encoding'] = encoding
        captured['errors'] = errors
        stderr_text = (
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'free-test/abc.mp4':\n"
            "  Duration: 00:00:05.00, start: 1.234000, bitrate: 1024 kb/s\n"
        )
        return subprocess.CompletedProcess(cmd, 0, None, stderr_text)

    monkeypatch.setattr(common.ffmpeg, 'get_ffmpeg_exe', fake_get_ffmpeg_exe)
    monkeypatch.setattr(common.subprocess, 'run', fake_run)

    start_time = common.get_start_time_ffmpeg('free-test/abc.mp4')

    assert start_time == pytest.approx(1.234)
    assert captured['cmd'] == ['/tmp/fake_ffmpeg', '-i', 'free-test/abc.mp4']
    assert captured['stderr'] == subprocess.PIPE
    assert captured['stdout'] == subprocess.DEVNULL
    assert captured['text'] is True
    assert captured['encoding'] == 'utf-8'
    assert captured['errors'] == 'replace'


def test_transcode_video_ffmpeg_builds_h264_command(monkeypatch, tmp_path):
    '''
    Verify ffmpeg command includes H.264 args when h264 codec is requested.
    '''

    from Sports2D.Utilities import common

    captured = {}

    def fake_get_ffmpeg_exe():
        return '/tmp/fake_ffmpeg'

    def fake_run(cmd, stdout=None, stderr=None, text=None, encoding=None, errors=None, check=None):
        captured['cmd'] = cmd
        captured['stdout'] = stdout
        captured['stderr'] = stderr
        captured['text'] = text
        captured['encoding'] = encoding
        captured['errors'] = errors
        captured['check'] = check
        return subprocess.CompletedProcess(cmd, 0, '', '')

    monkeypatch.setattr(common.ffmpeg, 'get_ffmpeg_exe', fake_get_ffmpeg_exe)
    monkeypatch.setattr(common.subprocess, 'run', fake_run)

    input_path = tmp_path / 'input.mp4'
    output_path = tmp_path / 'output.mp4'
    common.transcode_video_ffmpeg(
        input_path,
        output_path,
        codec='h264',
        source_fps=30.0,
        desired_framerate=25.0,
    )

    cmd = captured['cmd']
    assert cmd[0] == '/tmp/fake_ffmpeg'
    assert '-filter:v' in cmd
    assert 'setpts=1.2*PTS' in cmd
    assert '-r' in cmd
    assert '25.000000' in cmd
    codec_idx = cmd.index('-c:v')
    assert cmd[codec_idx + 1] == 'libx264'
    preset_idx = cmd.index('-preset')
    assert cmd[preset_idx + 1] == 'medium'
    crf_idx = cmd.index('-crf')
    assert cmd[crf_idx + 1] == '23'
    audio_codec_idx = cmd.index('-c:a')
    assert cmd[audio_codec_idx + 1] == 'aac'
    audio_bitrate_idx = cmd.index('-b:a')
    assert cmd[audio_bitrate_idx + 1] == '128k'
    movflags_idx = cmd.index('-movflags')
    assert cmd[movflags_idx + 1] == '+faststart'
    assert '-y' in cmd
    assert cmd[-1] == str(output_path)
    assert captured['stdout'] == subprocess.PIPE
    assert captured['stderr'] == subprocess.PIPE
    assert captured['text'] is True
    assert captured['encoding'] == 'utf-8'
    assert captured['errors'] == 'replace'
    assert captured['check'] is True


def test_resample_video_uses_mp4v_transcode_wrapper(monkeypatch, tmp_path):
    '''
    Verify resample helper delegates to ffmpeg transcode with mp4v codec.
    '''

    from Sports2D.Utilities import common

    source_path = tmp_path / 'demo.mp4'
    source_path.write_bytes(b'source')
    expected_tmp = tmp_path / 'demo_2.mp4'
    called = {}

    def fake_transcode(input_video_path, output_video_path, codec='h264',
                       source_fps=None, desired_framerate=None):
        called['input'] = Path(input_video_path)
        called['output'] = Path(output_video_path)
        called['codec'] = codec
        called['source_fps'] = source_fps
        called['desired_framerate'] = desired_framerate
        Path(output_video_path).write_bytes(b'converted')

    monkeypatch.setattr(common, 'transcode_video_ffmpeg', fake_transcode)

    common.resample_video(source_path, 30.0, 24.0)

    assert called['input'] == source_path
    assert called['output'] == expected_tmp
    assert called['codec'] == 'mp4v'
    assert called['source_fps'] == 30.0
    assert called['desired_framerate'] == 24.0
    assert source_path.read_bytes() == b'converted'
    assert not expected_tmp.exists()


def test_create_realtime_display_falls_back_to_opencv_when_qt_unavailable(monkeypatch):
    '''
    Verify qt backend request falls back to OpenCV backend when Qt import fails.
    '''

    from Sports2D.Utilities import realtime_display

    class DummyDisplay:
        backend_name = 'opencv'

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def raise_qt_import_error():
        raise ImportError('PySide6 missing')

    monkeypatch.setattr(realtime_display, '_load_qt_display_class', raise_qt_import_error)
    monkeypatch.setattr(realtime_display, 'OpenCVRealtimeDisplay', DummyDisplay)

    display = realtime_display.create_realtime_display(
        backend='qt',
        window_title='Test',
        display_width=640,
        display_height=480,
    )

    assert isinstance(display, DummyDisplay)


def test_expand_video_input_paths_supports_relative_directory(tmp_path):
    '''
    Verify relative directory input is expanded to sorted video files.
    '''

    from Sports2D.Sports2D import _expand_video_input_paths

    video_dir = tmp_path / 'videos'
    batch_dir = video_dir / 'batch'
    nested_dir = batch_dir / 'nested'
    nested_dir.mkdir(parents=True)

    (batch_dir / 'clip_b.MOV').write_text('')
    (batch_dir / 'clip_a.mp4').write_text('')
    (batch_dir / 'notes.txt').write_text('')
    (nested_dir / 'clip_nested.mp4').write_text('')

    video_files = _expand_video_input_paths('batch', video_dir)

    assert video_files == [Path('batch') / 'clip_a.mp4', Path('batch') / 'clip_b.MOV']


def test_expand_video_input_paths_supports_absolute_directory(tmp_path):
    '''
    Verify absolute directory input is expanded to sorted absolute video files.
    '''

    from Sports2D.Sports2D import _expand_video_input_paths

    batch_dir = tmp_path / 'batch'
    batch_dir.mkdir()

    (batch_dir / 'run_02.mp4').write_text('')
    (batch_dir / 'run_01.avi').write_text('')

    video_files = _expand_video_input_paths(str(batch_dir), tmp_path)

    assert video_files == [batch_dir / 'run_01.avi', batch_dir / 'run_02.mp4']


def test_expand_video_input_paths_raises_on_empty_directory(tmp_path):
    '''
    Verify directory input fails with a clear error when no videos are found.
    '''

    from Sports2D.Sports2D import _expand_video_input_paths

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match='No video files found'):
        _expand_video_input_paths('empty', tmp_path)


def test_workflow():
    '''
    Test the workflow of Sports2D.
    '''

    from Sports2D import Sports2D
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    #############################
    ## From Python             ##
    #############################

    # Default from the demo config file
    config_path = Path(__file__).resolve().parent.parent / 'Demo' / 'Config_demo.toml'
    config_dict = toml.load(config_path)
    video_dir = Path(__file__).resolve().parent.parent / 'Demo'
    config_dict.get("base").update({"video_dir": str(video_dir)})
    config_dict.get("base").update({"person_ordering_method": "highest_likelihood"})
    config_dict.get("base").update({"show_realtime_results":False})
    config_dict.get("post-processing").update({"show_graphs":False})
    config_dict.get("post-processing").update({"save_graphs":False})
    
    Sports2D.process(config_dict)


    # Only passing the updated values
    video_dir = Path(__file__).resolve().parent.parent / 'Demo'
    config_dict = {
      'base': {
        'nb_persons_to_detect': 1,
        'person_ordering_method': 'greatest_displacement',
        "show_realtime_results":False
        },
      'pose': {
        'mode': 'lightweight', 
        'det_frequency': 50
        },
      'post-processing': {
        'show_graphs':False,
        'save_graphs':False
        }
    }
    
    Sports2D.process(config_dict)


    #############################
    ## From command line (CLI) ##
    #############################

    # Default
    demo_cmd = ["sports2d", "--person_ordering_method", "highest_likelihood", "--show_realtime_results", "False", "--show_graphs", "False", "--save_graphs", "False"]
    subprocess.run(demo_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

    # With loading a trc file, visible_side 'front', first_person_height '1.76", floor_angle 0, xy_origin [0, 928]
    demo_cmd2 = ["sports2d", "--show_realtime_results", "False", "--show_graphs", "False", "--save_graphs", "False",
                 "--load_trc_px", os.path.join(root_dir, "demo_Sports2D", "demo_Sports2D_px_person01.trc"),
                 "--visible_side", "front", "--first_person_height", "1.76", "--time_range", "1.2", "2.7",
                 "--floor_angle", "0", "--xy_origin", "0", "928"]
    subprocess.run(demo_cmd2, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

    # With no pixels to meters conversion, one person to select, lightweight mode, detection frequency, slowmo factor, gaussian filter, RTMO body pose model
    demo_cmd3 = ["sports2d", "--show_realtime_results", "False", "--show_graphs", "False", "--save_graphs", "False",
                 "--floor_angle", "from_calib", "--xy_origin", "from_calib", "--perspective_unit", "from_calib", "--calib_file", os.path.join(root_dir, "demo_Sports2D", "demo_Sports2D_calib.toml"), 
                 "--nb_persons_to_detect", "1", "--person_ordering_method", "greatest_displacement", 
                 "--mode", "lightweight", "--det_frequency", "50", 
                 "--slowmo_factor", "4",
                 "--filter_type", "gaussian", "--use_augmentation", "False",
                 "--pose_model", "body", "--mode", """{'pose_class':'RTMO', 'pose_model':'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip', 'pose_input_size':[640, 640]}"""]
    subprocess.run(demo_cmd3, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    # With a time range, inverse kinematics, marker augmentation, perspective value in fov
    demo_cmd4 = ["sports2d", "--person_ordering_method", "greatest_displacement", "--show_realtime_results", "False", "--show_graphs", "False", "--save_graphs", "False",
                 "--time_range", "1.2", "2.7",
                 "--perspective_value", "40", "--perspective_unit", "fov_deg",
                 "--do_ik", "True", "--use_augmentation", "True", 
                 "--nb_persons_to_detect", "all", "--first_person_height", "1.65",
                 "--visible_side", "auto", "front", "--participant_mass", "55.0", "67.0"]
    subprocess.run(demo_cmd4, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    # From config file
    config_path = Path(__file__).resolve().parent.parent / 'Demo' / 'Config_demo.toml'
    config_dict = toml.load(config_path)
    video_dir = Path(__file__).resolve().parent.parent / 'Demo'
    config_dict.get("base").update({"video_dir": str(video_dir)})
    config_dict.get("base").update({"person_ordering_method": "highest_likelihood"})
    with open(config_path, 'w') as f: toml.dump(config_dict, f)
    demo_cmd5 = ["sports2d", "--config", str(config_path), "--show_realtime_results", "False", "--show_graphs", "False", "--save_graphs", "False",]
    subprocess.run(demo_cmd5, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')


if __name__ == "__main__":
    test_workflow()
