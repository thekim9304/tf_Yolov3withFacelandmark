import tensorflow as tf


def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_landmarks):
    """
    Given a cell offset prediction from the model, calculate the absolute box coordinates to the whole image.
    It's also an adpation of the original C code here:
    https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L83
    note that, we divide w and h by grid size

    inputs:
    y_pred: Prediction tensor from the model output, in the shape of (batch, grid, grid, anchor, 5 + num_classes)

    outputs:
    y_box: boxes in shape of (batch, grid, grid, anchor, 4), the last dimension is (xmin, ymin, xmax, ymax)
    objectness: probability that an object exists
    classes: probability of classes
    """
    box_xy, box_wh, objectness, landmarks = tf.split(y_pred, (2, 2, 1, num_landmarks), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    landmarks_probs = tf.sigmoid(landmarks)
    bbox_rel = tf.concat([box_xy, box_wh], axis=-1)

    grid_size = tf.shape(y_pred)[1]
    # meshgrid generates a grid that repeats by given range. It's the Cx and Cy in YoloV3 paper.
    # for example, tf.meshgrid(tf.range(3), tf.range(3)) will generate a list with two elements
    # note that in real code, the grid_size should be something like 13, 26, 52 for examples here and below
    #
    # [[0, 1, 2],
    #  [0, 1, 2],
    #  [0, 1, 2]]
    #
    # [[0, 0, 0],
    #  [1, 1, 1],
    #  [2, 2, 2]]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    # next, we stack two items in the list together in the last dimension, so that
    # we can interleve these elements together and become this:
    #
    # [[[0, 0], [1, 0], [2, 0]],
    #  [[0, 1], [1, 1], [2, 1]],
    #  [[0, 2], [1, 2], [2, 2]]]
    # let's add an empty dimension at axis=2 to expand the tensor to this:
    #
    # [[[[0, 0]], [[1, 0]], [[2, 0]]],
    #  [[[0, 1]], [[1, 1]], [[2, 1]]],
    #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
    #
    # at this moment, we now have a grid, which can always give us (y, x)
    # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    # YoloV2, YoloV3:
    # bx = sigmoid(tx) + Cx
    # by = sigmoid(ty) + Cy
    #
    # for example, if all elements in b_xy are (0.1, 0.2), the result will be
    #
    # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
    #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
    #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
    box_xy = box_xy + tf.cast(grid, tf.float32)
    # finally, divide this absolute box_xy by grid_size, and then we will get the normalized bbox centroids
    # for each anchor in each grid cell. b_xy is now in shape (batch_size, grid_size, grid_size, num_anchor, 2)
    #
    # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
    #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
    #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
    box_xy = box_xy / tf.cast(grid_size, tf.float32)

    # YoloV2:
    # "If the cell is offset from the top left corner of the image by (cx , cy)
    # and the bounding box prior has width and height pw , ph , then the predictions correspond to: "
    #
    # https://github.com/pjreddie/darknet/issues/568#issuecomment-469600294
    # "It’s OK for the predicted box to be wider and/or taller than the original image, but
    # it does not make sense for the box to have a negative width or height. That’s why
    # we take the exponent of the predicted number."
    box_wh = tf.exp(box_wh) * valid_anchors_wh

    bbox_abs = tf.concat([box_xy, box_wh], axis=-1)
    return bbox_abs, objectness, landmarks_probs, bbox_rel


def get_relative_yolo_box(y_true, valid_anchors_wh, num_landmarks):
    box_xy, box_wh, objectness, landmarks = tf.split(y_true, (2, 2, 1, num_landmarks), axis=-1)

    bbox_abs = tf.concat([box_xy, box_wh], axis=-1)

    grid_size = tf.shape(y_true)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)

    true_wh = tf.math.log(true_wh / valid_anchors_wh)
    true_wh = tf.where(
        tf.logical_or(tf.math.is_inf(true_wh), tf.math.is_nan(true_wh)),
        tf.zeros_like(true_wh), true_wh)

    bbox_rel = tf.concat([true_xy, true_wh], axis=-1)
    return bbox_rel, objectness, landmarks, bbox_abs


def xywh_to_x1x2y1y2(box):
    xy = box[..., 0:2]
    wh = box[..., 2:4]

    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2

    y_box = tf.concat([x1y1, x2y2], axis=-1)
    return y_box


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))
    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)

    # print(box_1.shape)
    # print(box_2.shape)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    # print(new_shape)

    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)

    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])

    return int_area / (box_1_area + box_2_area - int_area)
