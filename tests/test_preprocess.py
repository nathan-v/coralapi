"""LoadedModel input preprocessing: int8 quantization and NCHW layout.

These exercise the real LoadedModel._preprocess / input_size via a fake
interpreter (the `loaded_model` factory fixture in conftest), so no TPU needed.
"""

import numpy as np
from PIL import Image


def solid(w, h, rgb):
    return Image.new("RGB", (w, h), rgb)


# -- Bug 1: int8 input tensors must be quantized, not wrapped ----------------------


def test_int8_input_applies_zero_point_shift(loaded_model):
    # ultralytics int8 export quant params: scale ~ 1/255, zero_point -128.
    m = loaded_model([1, 2, 2, 3], np.int8, quantization=(1 / 255.0, -128))
    # (200 is >127; a naive astype(int8) would wrap it to -56 instead of shifting)
    out = m._preprocess(solid(2, 2, (200, 50, 255)))
    assert out.dtype == np.int8
    assert out.shape == (1, 2, 2, 3)
    # For (1/255, -128): quantized value == pixel - 128, per channel.
    np.testing.assert_array_equal(out[0, 0, 0], np.array([72, -78, 127], dtype=np.int8))
    assert out[0, 0, 0, 0] == 72  # shifted, not wrapped (-56)


def test_int8_input_clips_to_range(loaded_model):
    # A scale that would push a value past int8 max must clip, not overflow.
    m = loaded_model([1, 1, 1, 3], np.int8, quantization=(1 / 512.0, 0))
    out = m._preprocess(solid(1, 1, (255, 255, 255)))  # 255/255 / (1/512) = 512 -> 127
    assert (out == 127).all()


# -- Bug 2: NCHW input layout ------------------------------------------------------


def test_nchw_input_size_and_transpose(loaded_model):
    m = loaded_model([1, 3, 4, 6], np.uint8)  # NCHW [1, C=3, H=4, W=6]
    assert m.input_size == (6, 4)  # (W, H) read from dims 3, 2
    out = m._preprocess(solid(6, 4, (1, 2, 3)))
    assert out.shape == (1, 3, 4, 6)  # channels-first
    np.testing.assert_array_equal(out[0, :, 0, 0], np.array([1, 2, 3], dtype=np.uint8))


# -- Regression: uint8 / float32 NHWC unchanged (byte-identical) -------------------


def test_uint8_nhwc_is_bit_identical_to_prior_behavior(loaded_model):
    m = loaded_model([1, 4, 6, 3], np.uint8)  # non-square, to catch any W/H swap
    assert m.input_size == (6, 4)
    img = solid(6, 4, (10, 128, 240))
    out = m._preprocess(img)
    assert out.dtype == np.uint8
    assert out.shape == (1, 4, 6, 3)
    # Exactly what the old code produced: expand_dims of the resized RGB array.
    expected = np.expand_dims(np.asarray(img.convert("RGB").resize((6, 4), Image.LANCZOS)), axis=0)
    np.testing.assert_array_equal(out, expected)


def test_float32_nhwc_unchanged(loaded_model):
    m = loaded_model([1, 2, 2, 3], np.float32)
    out = m._preprocess(solid(2, 2, (0, 128, 255)))
    assert out.dtype == np.float32
    assert out.shape == (1, 2, 2, 3)
    np.testing.assert_allclose(out[0, 0, 0], np.array([0.0, 128 / 255, 1.0]), atol=1e-6)


# -- Routing: a single 3-D output engages the YOLO decoder, not SSD ---------------


def test_detect_routes_single_3d_output_to_yolo(loaded_model):
    out = np.zeros((1, 6, 8), dtype=np.float32)  # (1, 4+nc, N): 4 box + 2 classes
    out[0, :, 0] = [0.5, 0.5, 0.2, 0.2, 0.0, 0.9]  # anchor 0 -> class 1, score 0.9
    out_detail = {
        "index": 1,
        "dtype": np.float32,
        "quantization": (0.0, 0),
        "shape": np.array([1, 6, 8]),
    }
    m = loaded_model([1, 320, 320, 3], np.uint8, outputs=[(out_detail, out)], labels={1: "car"})
    results = m.detect(solid(320, 320, (0, 0, 0)), threshold=0.4)
    assert len(results) == 1
    assert results[0].index == 1 and results[0].label == "car"
    assert results[0].score > 0.8
