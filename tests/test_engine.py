import numpy as np

from coralapi.runtime.engine import decode_ssd_outputs, decode_yolo_output, load_labels


def test_load_labels_indexed(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("0 person\n1 bicycle\n2 car\n")
    assert load_labels(p) == {0: "person", 1: "bicycle", 2: "car"}


def test_load_labels_plain(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("background\ntench\ngoldfish\n")
    assert load_labels(p) == {0: "background", 1: "tench", 2: "goldfish"}


def test_load_labels_skips_blank_lines(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("cat\n\ndog\n")
    assert load_labels(p) == {0: "cat", 2: "dog"}


def yolo_head(anchors):
    """Build a (1, channels, num_anchors) YOLO head from per-anchor rows."""
    arr = np.zeros((len(anchors[0]), 8), dtype=np.float32)
    for i, anchor in enumerate(anchors):
        arr[:, i] = anchor
    return arr[None]


def test_decode_yolo_thresholds_and_nms():
    # channels = cx, cy, w, h, then 2 class scores; 8 anchor slots.
    anchors = [
        [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],  # kept: class 0
        [0.51, 0.5, 0.2, 0.2, 0.8, 0.0],  # suppressed by NMS (same object)
        [0.2, 0.2, 0.1, 0.1, 0.0, 0.3],  # dropped: below threshold
        [0.8, 0.8, 0.1, 0.1, 0.0, 0.7],  # kept: class 1
    ]
    labels = {0: "cat", 1: "dog"}
    dets = decode_yolo_output(yolo_head(anchors), labels, threshold=0.4)

    assert [(d.label, round(d.score, 2)) for d in dets] == [("cat", 0.9), ("dog", 0.7)]
    np.testing.assert_allclose(dets[0].box, [0.4, 0.4, 0.6, 0.6], atol=1e-6)


def test_decode_yolo_same_box_different_class_not_suppressed():
    anchors = [
        [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],
        [0.5, 0.5, 0.2, 0.2, 0.0, 0.8],  # same box, other class: kept
    ]
    dets = decode_yolo_output(yolo_head(anchors), {}, threshold=0.4)
    assert len(dets) == 2


def test_decode_yolo_accepts_transposed_layout():
    anchors = [[0.5, 0.5, 0.2, 0.2, 0.9, 0.0]]
    transposed = np.transpose(yolo_head(anchors), (0, 2, 1))
    dets = decode_yolo_output(transposed, {0: "cat"}, threshold=0.4)
    assert len(dets) == 1 and dets[0].label == "cat"


def test_decode_yolo_empty_when_nothing_clears_threshold():
    anchors = [[0.5, 0.5, 0.2, 0.2, 0.1, 0.05]]
    assert decode_yolo_output(yolo_head(anchors), {}, threshold=0.4) == []


# -- SSD postprocess, both output orderings ---------------------------------------

BOXES = np.array(
    [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]], dtype=np.float32
)


def ssd_outputs(order, classes, scores, count):
    b = BOXES[None]
    c = np.array(classes, dtype=np.float32)[None]
    s = np.array(scores, dtype=np.float32)[None]
    cnt = np.array([count], dtype=np.float32)
    # TF1 TFLite_Detection_PostProcess order vs TF2 Object Detection API order.
    return [b, c, s, cnt] if order == "tf1" else [b, s, c, cnt]


def test_decode_ssd_tf1_order():
    outs = ssd_outputs("tf1", classes=[17, 2, 5], scores=[0.9, 0.6, 0.1], count=3)
    dets = decode_ssd_outputs(outs, {17: "cat", 2: "car"}, threshold=0.5)
    assert [(d.index, d.label, round(d.score, 2)) for d in dets] == [
        (17, "cat", 0.9),
        (2, "car", 0.6),
    ]


def test_decode_ssd_tf2_reversed_order_matches_tf1():
    # The two tf2_ssd_* models emit (boxes, scores, classes, count); value-based
    # disambiguation must produce the exact same result as the TF1 order.
    labels = {17: "cat", 2: "car"}
    tf1 = decode_ssd_outputs(ssd_outputs("tf1", [17, 2, 5], [0.9, 0.6, 0.1], 3), labels, 0.5)
    tf2 = decode_ssd_outputs(ssd_outputs("tf2", [17, 2, 5], [0.9, 0.6, 0.1], 3), labels, 0.5)
    assert [(d.index, d.label, d.score) for d in tf1] == [(d.index, d.label, d.score) for d in tf2]
    assert [d.index for d in tf2] == [17, 2]  # class ids, not scores


def test_decode_ssd_ignores_garbage_padding_beyond_count():
    # Rows past `count` are NaN (as real models emit); they must not be examined.
    outs = ssd_outputs("tf2", classes=[3, np.nan, np.nan], scores=[0.8, np.nan, np.nan], count=1)
    dets = decode_ssd_outputs(outs, {3: "dog"}, threshold=0.5)
    assert [(d.index, d.label) for d in dets] == [(3, "dog")]


def test_decode_ssd_binary_model_disambiguated_by_integrality():
    # Both tensors within [0, 1]; the integer-valued one is the class ids.
    outs = ssd_outputs("tf2", classes=[1, 0], scores=[0.95, 0.7], count=2)
    dets = decode_ssd_outputs(outs, {0: "bg", 1: "face"}, threshold=0.5)
    assert [(d.index, round(d.score, 2)) for d in dets] == [(1, 0.95), (0, 0.7)]
