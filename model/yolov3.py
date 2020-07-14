import sys
import time
sys.path.append('C:/Users/th_k9/Desktop/Yolov3')

from tensorflow.keras.layers import (
    Concatenate,
    Lambda,
    UpSampling2D,
)

from model.backbone.Darknet53 import *
from utils.utils import *

from utils.preprocess import Preprocessor, anchors_wh, anchors_wh_mask
from utils.postprocess import Postprocessor

def YoloV3(input_shape=(416, 416, 3), num_landmarks=196, training=False):
    # 3 * (4 + 1 + num_classes)
    # 3_scale * (box_coord + obj_true&false + num_classes)
    final_filters = 3 * (4 + 1 + num_landmarks)

    inputs = Input(shape=input_shape)

    backbone = Darknet(input_shape)
    x_small, x_medium, x_large = backbone(inputs)

    # large scale detection
    x = DarknetConv(
        x_large,
        512,
        kernel_size=1,
        strides=1,
        name='detector_scale_large_1x1_1')
    x = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_1')
    x = DarknetConv(
        x, 512, kernel_size=1, strides=1, name='detector_scale_large_1x1_2')
    x = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_2')
    x = DarknetConv(
        x, 512, kernel_size=1, strides=1, name='detector_scale_large_1x1_3')

    y_large = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_3')
    y_large = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_large_final_conv2d',
    )(y_large)

    # medium scale detection
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_0')
    x = UpSampling2D(size=(2, 2), name='detector_scale_1_upsampling')(x)
    x = Concatenate(name='detector_scale_1_concat')([x, x_medium])

    # print('medium_sclae {}'.format(x.shape))

    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_1')
    x = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_1')
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_2')
    x = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_2')
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_3')

    y_medium = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_3')
    y_medium = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_medium_final_conv2d',
    )(y_medium)

    # small scale detection
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_0')
    x = UpSampling2D(size=(2, 2), name='detector_scale_small_upsampling')(x)
    x = Concatenate(name='detector_scale_small_concat')([x, x_small])

    # print('small_sclae {}'.format(x.shape))

    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_1')
    x = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_1')
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_2')
    x = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_2')
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_3')

    y_small = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_3')
    y_small = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_small_final_conv2d',
    )(y_small)

    y_small_shape = tf.shape(y_small)
    y_medium_shape = tf.shape(y_medium)
    y_large_shape = tf.shape(y_large)

    y_small = tf.reshape(
        y_small, (y_small_shape[0], y_small_shape[1], y_small_shape[2], 3, -1),
        name='detector_reshape_small')
    y_medium = tf.reshape(
        y_medium,
        (y_medium_shape[0], y_medium_shape[1], y_medium_shape[2], 3, -1),
        name='detector_reshape_meidum')
    y_large = tf.reshape(
        y_large, (y_large_shape[0], y_large_shape[1], y_large_shape[2], 3, -1),
        name='detector_reshape_large')

    if training:
        return tf.keras.Model(inputs, (y_small, y_medium, y_large))

    box_small = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[0:3], num_landmarks),
        name='detector_final_box_small')(y_small)
    box_medium = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[3:6], num_landmarks),
        name='detector_final_box_medium')(y_medium)
    box_large = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[6:9], num_landmarks),
        name='detector_final_box_large')(y_large)

    outputs = (box_small, box_medium, box_large)
    return tf.keras.Model(inputs, outputs)


class YoloLoss:
    def __init__(self, valid_anchors_wh, num_landmarks, ignore_thresh=0.5, lambda_coord=5.0, lambda_noobj=0.5):
        self.valid_anchors_wh = valid_anchors_wh
        self.num_landmarks = num_landmarks
        self.ignore_thresh = ignore_thresh
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def __call__(self, y_true, y_pred):
        """
            - y_pred to bbox_abs
            - get pred_xy_rel and pred_wh_rel
        """
        pred_box_abs, pred_obj, pred_landmark, pred_box_rel = get_absolute_yolo_box(y_pred,
                                                                                    self.valid_anchors_wh,
                                                                                    self.num_landmarks)
        # pred_box_abs = (xywh)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)
        pred_xy_rel = pred_box_rel[..., 0:2]
        pred_wh_rel = pred_box_rel[..., 2:4]

        """
            - y_true to bbox_rel
            - get true_xy_rel and true_wh_rel
        """
        true_box_rel, true_obj, true_landmark, true_box_abs = get_relative_yolo_box(y_true,
                                                                                    self.valid_anchors_wh,
                                                                                    self.num_landmarks)

        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        true_wh_abs = true_box_abs[..., 2:4]
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        xy_loss = self.calc_xy_loss(true_xy_rel, pred_xy_rel, true_obj, weight)
        wh_loss = self.calc_xy_loss(true_wh_rel, pred_wh_rel, true_obj, weight)
        # !!**!! add landmark loss later
        landmark_loss = self.calc_xy_loss(true_landmark, pred_landmark, true_obj, weight)

        # use the absolute yolo box to calculate iou and ignore mask
        ignore_mask = self.calc_ignore_mask(true_box_abs, pred_box_abs, true_obj)

        # print('=' * 10, 'xy_loss', '=' * 10)
        # print(xy_loss)
        # print('=' * 10, 'wh_loss', '=' * 10)
        # print(wh_loss)
        # print('=' * 10, 'landmark_loss', '=' * 10)
        # print(landmark_loss)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        return xy_loss + wh_loss + landmark_loss + obj_loss, (xy_loss, wh_loss, landmark_loss, obj_loss)

    def calc_xy_loss(self, true_xy, pred_xy, true_obj, weight):
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        xy_loss = xy_loss * true_obj * weight
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord

        return xy_loss

    def calc_ignore_mask(self, true_box, pred_box, true_obj):
        obj_mask = tf.squeeze(true_obj, -1)

        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask), tf.float32)

        ignore_mask = tf.cast(best_iou < self.ignore_thresh, tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        obj_entropy = self.binary_cross_entropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy

        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(noobj_loss, axis=(1, 2, 3, 4)) * self.lambda_noobj

        return obj_loss + noobj_loss

    def binary_cross_entropy(self, logits, labels):
        epsilon = 1e-7
        logits = tf.clip_by_value(logits, epsilon, 1 - epsilon)

        return -(labels * tf.math.log(logits) +
                 (1 - labels) * tf.math.log(1 - logits))


def main():
    # Prepare dataset(imgs, labels)
    preprocess = Preprocessor('C:/Users/th_k9/Desktop/Yolov3withFacelandmark/annotation_preparation/300VW_train', batch_size=2)

    imgs, labels = next(preprocess())
    #
    # print(imgs.shape)
    # print(labels[0].shape)
    # print(labels[1].shape)
    # print(labels[2].shape)

    # Training
    yolov3 = YoloV3(input_shape=(416, 416, 3), num_landmarks=136, training=True)
    outputs = yolov3(imgs)

    # loss_func1 = YoloLoss(anchors_wh_mask[0], 136)
    loss_func2 = YoloLoss(anchors_wh_mask[1], 136)
    # loss_func3 = YoloLoss(anchors_wh_mask[2], 136)

    # loss1, loss_breakdown1 = loss_func1(labels[0], outputs[0])
    loss2, loss_breakdown2 = loss_func2(labels[1], outputs[1])
    # loss3, loss_breakdown3 = loss_func3(labels[2], outputs[2])

    # print(loss1)
    # print(loss2)
    # print(loss3)

    # Inference
    # yolov3 = YoloV3(input_shape=(416, 416, 3), num_classes=num_classes, training=False)
    # outputs = yolov3(imgs)
    #
    # postprocess = Postprocessor(0.5, 0.5, 3)
    # boxes, scores, classes, num_detection = postprocess(outputs)
    #
    # print(boxes)
    # print(scores)
    # print(classes)
    # print(num_detection)


if __name__ == '__main__':
    main()
