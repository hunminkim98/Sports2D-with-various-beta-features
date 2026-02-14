import pytest

def _load_deps():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from Sports2D.Utilities.pose3d_adapter import (
        H36M_17_JOINT_NAMES,
        prepare_fmpose3d_input_from_xy,
    )
    return np, pd, H36M_17_JOINT_NAMES, prepare_fmpose3d_input_from_xy


def _build_xy_dataframe(np, pd, num_frames: int = 8):
    t = np.arange(num_frames, dtype=np.float32)
    base = {
        "Nose": 100 + t,
        "LEye": 95 + t,
        "REye": 105 + t,
        "LEar": 90 + t,
        "REar": 110 + t,
        "LShoulder": 80 + t,
        "RShoulder": 120 + t,
        "LElbow": 75 + t,
        "RElbow": 125 + t,
        "LWrist": 70 + t,
        "RWrist": 130 + t,
        "LHip": 85 + t,
        "RHip": 115 + t,
        "LKnee": 85 + t * 0.9,
        "RKnee": 115 + t * 0.9,
        "LAnkle": 85 + t * 0.7,
        "RAnkle": 115 + t * 0.7,
        # Extra foot markers that must be ignored by 3D adapter
        "LBigToe": 90 + t * 0.5,
        "RBigToe": 110 + t * 0.5,
        "LHeel": 88 + t * 0.5,
        "RHeel": 112 + t * 0.5,
    }
    x_df = pd.DataFrame(base)
    y_df = pd.DataFrame({k: v + 50 for k, v in base.items()})
    return x_df, y_df


def test_prepare_fmpose3d_input_shapes_and_no_nan():
    np, pd, H36M_17_JOINT_NAMES, prepare_fmpose3d_input_from_xy = _load_deps()
    x_df, y_df = _build_xy_dataframe(np, pd, num_frames=12)
    h36m_norm, valid_frames, missing = prepare_fmpose3d_input_from_xy(
        x_df=x_df,
        y_df=y_df,
        cam_width=1280,
        cam_height=720,
    )

    assert h36m_norm.shape == (12, 17, 2)
    assert np.isfinite(h36m_norm).all()
    assert len(valid_frames) > 0
    assert isinstance(missing, list)
    assert len(H36M_17_JOINT_NAMES) == 17


def test_prepare_fmpose3d_input_handles_missing_face_points():
    np, pd, _, prepare_fmpose3d_input_from_xy = _load_deps()
    x_df, y_df = _build_xy_dataframe(np, pd, num_frames=6)
    x_df = x_df.drop(columns=["LEye", "REye", "LEar", "REar"])
    y_df = y_df.drop(columns=["LEye", "REye", "LEar", "REar"])

    h36m_norm, valid_frames, missing = prepare_fmpose3d_input_from_xy(
        x_df=x_df,
        y_df=y_df,
        cam_width=640,
        cam_height=480,
    )

    assert h36m_norm.shape == (6, 17, 2)
    assert np.isfinite(h36m_norm).all()
    assert "LEye" in missing and "REye" in missing
    assert len(valid_frames) > 0


def test_prepare_fmpose3d_input_mirror_fallback_for_missing_left_leg():
    np, pd, _, prepare_fmpose3d_input_from_xy = _load_deps()
    x_df, y_df = _build_xy_dataframe(np, pd, num_frames=6)
    x_df = x_df.drop(columns=["LHip", "LKnee", "LAnkle"])
    y_df = y_df.drop(columns=["LHip", "LKnee", "LAnkle"])

    h36m_norm, valid_frames, missing = prepare_fmpose3d_input_from_xy(
        x_df=x_df,
        y_df=y_df,
        cam_width=640,
        cam_height=480,
    )

    assert h36m_norm.shape == (6, 17, 2)
    assert np.isfinite(h36m_norm).all()
    assert "LHip" in missing and "LKnee" in missing and "LAnkle" in missing
    assert len(valid_frames) > 0
