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
import os
import toml
import subprocess
from pathlib import Path
import numpy as np
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


def test_parse_ball_ordering_method_falls_back_for_invalid_values():
    '''
    Verify invalid ball ordering methods fall back to the provided default.
    '''

    from Sports2D.process import _parse_ball_ordering_method

    assert _parse_ball_ordering_method('largest_size') == 'largest_size'
    assert _parse_ball_ordering_method('not_a_mode', default='first_detected') == 'first_detected'


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


def test_transcode_video_ffmpeg_builds_h264_command(monkeypatch, tmp_path):
    '''
    Verify ffmpeg command includes H.264 args when h264 codec is requested.
    '''

    from Sports2D.Utilities import common

    captured = {}

    def fake_get_ffmpeg_exe():
        return '/tmp/fake_ffmpeg'

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=None):
        captured['cmd'] = cmd
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
