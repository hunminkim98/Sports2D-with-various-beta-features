import pytest

def _load_inference_api():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from Sports2D.FMPose3D.inference import prepare_fmpose3d_input, resolve_device
    return np, prepare_fmpose3d_input, resolve_device


def test_prepare_fmpose3d_input_tensor_shape():
    np, prepare_fmpose3d_input, _ = _load_inference_api()
    keypoints = np.random.rand(10, 17, 2).astype(np.float32)
    model_input = prepare_fmpose3d_input(keypoints, augment=True)
    assert tuple(model_input.shape) == (1, 2, 10, 17, 2)


def test_resolve_device_cpu():
    _, _, resolve_device = _load_inference_api()
    device = resolve_device("cpu")
    assert str(device) == "cpu"
