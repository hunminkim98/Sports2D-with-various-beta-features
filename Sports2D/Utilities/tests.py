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
