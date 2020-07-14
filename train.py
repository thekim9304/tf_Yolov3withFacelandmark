import os
import datetime

import tensorflow as tf
from tensorflow.keras.layers import Lambda

from model.yolov3 import YoloV3, YoloLoss
from utils.preprocess import Preprocessor, anchors_wh_mask, anchors_wh
from utils.postprocess import Postprocessor
from utils.utils import get_absolute_yolo_box

BATCH_SIZE = 8
EPOCH = 1000

def main():
    data_dir = './annotation_preparation/33'
    ckpt_dir = 'E:/checkpoints'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    num_landmarks = 136
    lr_rate = 0.0001
    pre_train = False

    preprocessor = Preprocessor(data_dir=data_dir,
                                output_shape=(416, 416),
                                batch_size=BATCH_SIZE)
    postprocessor = Postprocessor(0.5, 0.5, 3)

    model = YoloV3(input_shape=(416, 416, 3), num_landmarks=num_landmarks, training=True)
    if pre_train:
        ckpt_path = 'E:/checkpoints'
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        print('{} load done!'.format(ckpt))
        model.load_weights(ckpt)

        init_epoch = int(os.path.split(str(ckpt))[-1].split('-')[1]) + 1
        lowest_loss = float(os.path.split(str(ckpt))[-1].split('-')[2].replace('.ckpt', ''))
    else:
        init_epoch = 0
        lowest_loss = 2

    print('init_epoch : {}'.format(init_epoch))
    print('lowest_loss : {}'.format(lowest_loss))

    loss_objects = [YoloLoss(valid_anchors_wh, num_landmarks) for valid_anchors_wh in anchors_wh_mask]
    optimizer = tf.keras.optimizers.Adam(lr=lr_rate)

    saved = False
    cnt_to_early = 0
    for epoch in range(init_epoch, EPOCH + init_epoch, 1):
        print()
        print('{} epoch start! : {}'.format(epoch, datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")))

        epoch_loss = train_one_epoch(model, loss_objects, preprocessor(), optimizer, postprocessor)

        if lowest_loss > epoch_loss:
            lowest_loss = epoch_loss

            save_path = ckpt_dir + '/cp-{:04d}-{:.4f}.ckpt'.format(epoch, lowest_loss)
            model.save_weights(save_path)
            print('Save CKPT _ [loss : {:.4f}, save_path : {}]\n'.format(lowest_loss, save_path))

            cnt_to_early = 0
            saved = True

        if saved:
            if cnt_to_early >= 30:
                lr_rate /= 10
                cnt_to_early = 0
                print('=============Down the learning rate : {}============='.format(lr_rate))

            cnt_to_early += 1
            if lr_rate < 0.000001 and cnt_to_early >= 30:
                print('Early Stopped on {} Epoch and {} Loss.'.format(epoch, lowest_loss))
                break


def train_one_epoch(model, loss_objects, generator, optimizer, post):
    epoch_total_loss = 0.0
    epoch_xy_loss = 0.0
    epoch_wh_loss = 0.0
    epoch_landmark_loss = 0.0
    epoch_obj_loss = 0.0
    batchs = 0

    for i, (images, labels) in enumerate(generator):
        batch_size = images.shape[0]
        with tf.GradientTape() as tape:

            outputs = model(images)
            '''
            '''
            box_small = Lambda(
                lambda x: get_absolute_yolo_box(x, anchors_wh[0:3], 136),
                name='detector_final_box_small')(outputs[0])
            box_medium = Lambda(
                lambda x: get_absolute_yolo_box(x, anchors_wh[3:6], 136),
                name='detector_final_box_medium')(outputs[1])
            box_large = Lambda(
                lambda x: get_absolute_yolo_box(x, anchors_wh[6:9], 136),
                name='detector_final_box_large')(outputs[2])

            print(labels[1][0][8][16][0][4])
            print(box_medium[1][0][8][16][0])

            # post((box_small, box_medium, box_large))
            '''
            '''

            total_losses, xy_losses, wh_losses, landmark_losses, obj_losses = [], [], [], [], []

            # iterate over all three sclaes
            for loss_object, y_pred, y_true in zip(loss_objects, outputs, labels):
                total_loss, loss_breakdown = loss_object(y_true, y_pred)
                xy_loss, wh_loss, landmark_loss, obj_loss = loss_breakdown
                total_losses.append(total_loss * (1. / batch_size))
                xy_losses.append(xy_loss * (1. / batch_size))
                wh_losses.append(wh_loss * (1. / batch_size))
                landmark_losses.append(landmark_loss * (1. / batch_size))
                obj_losses.append(obj_loss * (1. / batch_size))

            total_loss = tf.reduce_sum(total_losses)
            total_xy_loss = tf.reduce_sum(xy_losses)
            total_wh_loss = tf.reduce_sum(wh_losses)
            total_landmark_loss = tf.reduce_sum(landmark_losses)
            total_obj_loss = tf.reduce_sum(obj_losses)

        grads = tape.gradient(target=total_loss, sources=model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_total_loss += total_loss
        epoch_xy_loss += total_xy_loss
        epoch_wh_loss += total_wh_loss
        epoch_landmark_loss += total_landmark_loss
        epoch_obj_loss += total_obj_loss
        batchs += 1

    epoch_total_loss = epoch_total_loss / batchs
    # print(' total_loss.[{0:.4f}]'.format(epoch_total_loss))
    # print('     xy:{:.4f}, wh:{:.4f}, landmark:{:.4f}, obj:{:.4f}'.format(epoch_xy_loss / batchs,
    #                                                                    epoch_wh_loss / batchs,
    #                                                                    epoch_landmark_loss / batchs,
    #                                                                    epoch_obj_loss / batchs))
    print('objectness loss : {:.4}'.format(epoch_obj_loss/batchs))

    return epoch_total_loss

if __name__ == '__main__':
    main()
