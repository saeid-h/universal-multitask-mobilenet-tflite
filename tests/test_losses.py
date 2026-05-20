"""Tests for the upgraded YOLO and TextDetection loss implementations."""

import numpy as np
import pytest
import tensorflow as tf

from src.utils.losses import YOLOLoss, TextDetectionLoss, get_loss_for_head_type


# ---------------- YOLOLoss ----------------

def _yolo_targets(batch=2, h=4, w=4, num_boxes=1, num_classes=3, seed=0):
    """Build (y_true, y_pred) for a YOLO grid with a couple of positive cells."""
    rng = np.random.RandomState(seed)
    per_box = 5 + num_classes
    shape = (batch, h, w, num_boxes * per_box)
    y_true = np.zeros(shape, dtype=np.float32)
    # Put one positive object in batch 0 at cell (1, 2), and one in batch 1 at (3, 0).
    flat = y_true.reshape(batch, h, w, num_boxes, per_box)
    flat[0, 1, 2, 0, 0:4] = [0.3, 0.4, 0.5, 0.5]  # tx, ty, tw, th
    flat[0, 1, 2, 0, 4] = 1.0                       # objectness
    flat[0, 1, 2, 0, 5 + 1] = 1.0                   # class 1 one-hot
    flat[1, 3, 0, 0, 0:4] = [0.7, 0.1, 0.2, 0.8]
    flat[1, 3, 0, 0, 4] = 1.0
    flat[1, 3, 0, 0, 5 + 2] = 1.0                   # class 2 one-hot
    # y_pred is a small random perturbation around y_true (still in logit space).
    y_pred = y_true + rng.randn(*shape).astype(np.float32) * 0.1
    return tf.constant(y_true), tf.constant(y_pred)


def test_yolo_loss_basic_call():
    yt, yp = _yolo_targets(num_classes=3)
    loss = YOLOLoss(num_classes=3, num_boxes=1)
    out = loss(yt, yp).numpy()
    assert np.isfinite(out)
    assert out > 0


def test_yolo_loss_perfect_pred_lower_than_random():
    # Perfect predictions of objectness and class logits (large logits in
    # the correct direction) should give lower loss than random predictions.
    yt, _ = _yolo_targets(num_classes=3, seed=1)
    yt_np = yt.numpy()
    # Build a perfect-ish prediction: huge positive logits where target=1.
    yp_np = yt_np.copy()
    per_box = 5 + 3
    yp = yp_np.reshape(yp_np.shape[0], yp_np.shape[1], yp_np.shape[2], 1, per_box)
    # Boost objectness logit at object cells; pre-sigmoid xy maps to target via sigmoid(big).
    obj_mask = yp[..., 4:5] > 0.5
    yp[..., 4:5] = np.where(obj_mask, 5.0, -5.0)
    # Class logits: amplify the one-hot
    one_hot = (yt_np.reshape(*yp.shape)[..., 5:] > 0.5).astype(np.float32)
    yp[..., 5:] = one_hot * 5.0 - (1 - one_hot) * 5.0
    yp = yp.reshape(yp_np.shape)

    loss = YOLOLoss(num_classes=3, num_boxes=1)
    perfect = loss(tf.constant(yt_np), tf.constant(yp)).numpy()
    random = loss(tf.constant(yt_np), tf.constant(np.random.RandomState(2).randn(*yp_np.shape).astype(np.float32))).numpy()
    assert perfect < random, f"perfect={perfect} should be < random={random}"


def test_yolo_loss_rejects_bad_args():
    with pytest.raises(ValueError, match="num_classes must be"):
        YOLOLoss(num_classes=0)
    with pytest.raises(ValueError, match="num_boxes must be"):
        YOLOLoss(num_classes=1, num_boxes=0)


def test_yolo_loss_get_config_roundtrip():
    loss = YOLOLoss(num_classes=4, num_boxes=2, lambda_coord=3.0, lambda_noobj=0.7)
    config = loss.get_config()
    assert config['num_classes'] == 4
    assert config['num_boxes'] == 2
    assert config['lambda_coord'] == 3.0
    assert config['lambda_noobj'] == 0.7


def test_yolo_loss_via_get_loss_for_head_type_default_num_classes():
    # Should not raise: helper defaults num_classes to 1 when not provided.
    loss = get_loss_for_head_type('yolo_detection')
    assert isinstance(loss, YOLOLoss)
    assert loss.num_classes == 1


# ---------------- TextDetectionLoss ----------------

def _text_targets(batch=2, h=8, w=8, with_angle=False, seed=0):
    rng = np.random.RandomState(seed)
    ch = 6 if with_angle else 5
    y_true = np.zeros((batch, h, w, ch), dtype=np.float32)
    # A small text region in the middle of batch 0.
    y_true[0, 3:5, 3:5, 0] = 1.0  # score
    y_true[0, 3:5, 3:5, 1:5] = [0.1, 0.2, 0.3, 0.4]  # box geometry (dx,dy,dw,dh)
    if with_angle:
        y_true[0, 3:5, 3:5, 5] = 0.2
    # Pred is target + small noise; pre-sigmoid not relevant here (score is in [0,1]).
    y_pred = y_true + rng.randn(*y_true.shape).astype(np.float32) * 0.05
    y_pred[..., 0:1] = np.clip(y_pred[..., 0:1], 0.0, 1.0)  # keep score in [0,1]
    return tf.constant(y_true), tf.constant(y_pred)


def test_text_loss_tensor_input_no_angle():
    yt, yp = _text_targets(with_angle=False)
    out = TextDetectionLoss()(yt, yp).numpy()
    assert np.isfinite(out)
    assert out > 0


def test_text_loss_tensor_input_with_angle():
    yt, yp = _text_targets(with_angle=True)
    out = TextDetectionLoss(angle_weight=0.5)(yt, yp).numpy()
    assert np.isfinite(out)


def test_text_loss_dict_input():
    yt_np = np.zeros((1, 4, 4, 5), dtype=np.float32)
    yt_np[0, 1, 1, 0] = 1.0
    yt_np[0, 1, 1, 1:5] = [0.1, 0.2, 0.3, 0.4]
    yp_np = yt_np + np.random.RandomState(3).randn(*yt_np.shape).astype(np.float32) * 0.05
    yp_np[..., 0:1] = np.clip(yp_np[..., 0:1], 0.0, 1.0)

    yt_dict = {'text_scores': tf.constant(yt_np[..., 0:1]),
               'text_boxes':  tf.constant(yt_np[..., 1:5])}
    yp_dict = {'text_scores': tf.constant(yp_np[..., 0:1]),
               'text_boxes':  tf.constant(yp_np[..., 1:5])}
    out = TextDetectionLoss()(yt_dict, yp_dict).numpy()
    assert np.isfinite(out)


def test_text_loss_no_positive_pixels_is_finite():
    # All-zero score map -> geometry mask is zero everywhere. Loss must
    # not produce NaN (the epsilon in the denominator guards this).
    y = tf.zeros((1, 4, 4, 5), dtype=tf.float32)
    out = TextDetectionLoss()(y, y).numpy()
    assert np.isfinite(out)


def test_text_loss_perfect_pred_lower_than_random():
    yt, _ = _text_targets(with_angle=True, seed=5)
    perfect = TextDetectionLoss()(yt, yt).numpy()
    yp_random = tf.constant(np.random.RandomState(6).rand(*yt.shape).astype(np.float32))
    random = TextDetectionLoss()(yt, yp_random).numpy()
    assert perfect < random
