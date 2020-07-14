import os
import cv2
import time
import tensorflow as tf

from utils.postprocess import Postprocessor
from model.yolov3 import YoloV3

# Video
# def main():
#     # Setting
#     input_shape = (416, 416, 3)
#     num_landmarks = 136
#
#     # Load the trained model
#     model = YoloV3(input_shape=input_shape, num_landmarks=num_landmarks, training=False)
#     ckpt_path = 'C:/Users/th_k9/Desktop/Yolov3withFacelandmark/trained_weights/300VW'
#     ckpt = tf.train.latest_checkpoint(ckpt_path)
#     print('{} load done!'.format(ckpt))
#     model.load_weights(ckpt)
#
#     # Postprocessor object
#     postprocessor = Postprocessor(0.5, 0.5, 1)
#
#     # video_path = 0
#     video_path = 'E:/DB_FaceLandmark/300VW/001/vid.avi'
#     cap = cv2.VideoCapture(video_path)
#
#     prevTime = 0
#     while True:
#         ret, frame = cap.read()
#
#         if ret:
#             img = cv2.resize(frame, input_shape[:2])
#             x = cv2.resize(frame, input_shape[:2])
#             x = tf.expand_dims(x.astype('float32') / 255, axis=0)
#
#             # Prediction
#             y_pred = model(x)
#
#             # Postprocess
#             boxes, scores, landmarks, num_detection = postprocessor(y_pred)
#             landmarks = landmarks * input_shape[0]
#
#             num_img = num_detection.shape[0]
#             for img_i in range(num_img):
#                 # based on train data
#                 # h, w, d = x[img_i].shape
#                 # based on original image
#                 h, w, d = img.shape
#
#                 for i in range(num_detection[img_i][0].numpy()):
#                     box = boxes[img_i][i].numpy()
#
#                     xmin = int(box[0] * w)
#                     ymin = int(box[1] * h)
#                     xmax = int(box[2] * w)
#                     ymax = int(box[3] * h)
#
#                     img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#
#                 # for i in range(0, 136, 2):
#                 #     img = cv2.circle(img, (int(landmarks[0][img_i][i]), int(landmarks[0][img_i][i+1])), 2, (0, 0, 255), -1)
#
#             curTime = time.time()
#             sec = curTime - prevTime
#             print(sec)
#             prevTime = curTime
#             fps = 1/(sec)
#             print('fps : {}'.format(fps))
#
#             cv2.imshow('t', img)
#
#             if cv2.waitKey(1) == 27:
#                 break
#         else:
#             return
#     cv2.destroyAllWindows()
#     cap.release()

# Multi frame
# def main():
#     # Setting
#     input_shape = (416, 416, 3)
#     num_landmarks = 136
#
#     # Load the trained model
#     model = YoloV3(input_shape=input_shape, num_landmarks=num_landmarks, training=False)
#     ckpt_path = 'C:/Users/th_k9/Desktop/Yolov3withFacelandmark/trained_weights/300VW'
#     ckpt = tf.train.latest_checkpoint(ckpt_path)
#     print('{} load done!'.format(ckpt))
#     model.load_weights(ckpt)
#
#     # Postprocessor object
#     postprocessor = Postprocessor(0.5, 0.5, 1)
#
#     file_path = 'E:/DB_FaceLandmark/300VW_frame/test'
#     file_list = os.listdir(file_path)
#
#     save_path = 'E:/DB_FaceLandmark/300VW_frame/inference_test'
#
#     for file_name in file_list:
#         frame = cv2.imread(os.path.join(file_path, file_name))
#
#         img = cv2.resize(frame, input_shape[:2])
#         x = cv2.resize(frame, input_shape[:2])
#         x = tf.expand_dims(x.astype('float32') / 255, axis=0)
#
#         # Prediction
#         y_pred = model(x)
#
#         # Postprocess
#         boxes, scores, landmarks, num_detection = postprocessor(y_pred)
#         landmarks = landmarks * input_shape[0]
#
#         num_img = num_detection.shape[0]
#         for img_i in range(num_img):
#             # based on train data
#             # h, w, d = x[img_i].shape
#             # based on original image
#             h, w, d = img.shape
#
#             for i in range(num_detection[img_i][0].numpy()):
#                 box = boxes[img_i][i].numpy()
#
#                 xmin = int(box[0] * w)
#                 ymin = int(box[1] * h)
#                 xmax = int(box[2] * w)
#                 ymax = int(box[3] * h)
#
#                 img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#
#             for i in range(0, 136, 2):
#                 img = cv2.circle(img, (int(landmarks[0][img_i][i]), int(landmarks[0][img_i][i+1])), 1, (255, 255, 255), -1)
#
#         cv2.imwrite(os.path.join(save_path, file_name), img)
#         # cv2.imshow('t', img)
#         # cv2.waitKey()
#     # cv2.destroyAllWindows()


# Single frame
def main():
    # Setting
    input_shape = (416, 416, 3)
    num_landmarks = 136

    # Load the trained model
    model = YoloV3(input_shape=input_shape, num_landmarks=num_landmarks, training=False)
    ckpt_path = 'C:/Users/th_k9/Desktop/Yolov3withFacelandmark/trained_weights/300VW'
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    print('{} load done!'.format(ckpt))
    model.load_weights(ckpt)

    # # Prepare a input image
    # img_path = 'E:/DB_FaceLandmark/WFLW/WFLW_images/51--Dresses/51_Dresses_wearingdress_51_377.jpg'
    img_path = 'E:/DB_FaceLandmark/300VW_frame/train/001_0000.jpg'
    img = cv2.resize(cv2.imread(img_path), input_shape[:2])
    # img = cv2.resize(cv2.imread('E:/DB_FaceLandmark/WFLW/WFLW_images/51--Dresses/51_Dresses_wearingdress_51_377.jpg'), (832,832))
    # cv2.imshow('in_img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    x = tf.expand_dims(img.astype('float32') / 255, axis=0)

    # Postprocessor object
    postprocessor = Postprocessor(0.5, 0.5, 3)

    prevTime = time.time()

    # Predict
    y_pred = model(x)

    boxes, scores, landmarks, num_detection = postprocessor(y_pred)
    #
    # # print(landmarks)
    # landmarks = landmarks * img.shape[0]
    #
    # num_img = num_detection.shape[0]
    # for img_i in range(num_img):
    #     # based on train data
    #     # h, w, d = x[img_i].shape
    #     # based on original image
    #     h, w, d = img.shape
    #
    #     for i in range(num_detection[img_i][0].numpy()):
    #         box = boxes[img_i][i].numpy()
    #
    #         xmin = int(box[0] * w)
    #         ymin = int(box[1] * h)
    #         xmax = int(box[2] * w)
    #         ymax = int(box[3] * h)
    #
    #         img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #
    #     for i in range(0, 136, 2):
    #         img = cv2.circle(img, (int(landmarks[0][img_i][i]), int(landmarks[0][img_i][i+1])), 2, (0, 0, 255), -1)
    #
    # cv2.imshow('t', img)
    # # cv2.imwrite('C:/Users/th_k9/Desktop/fl_model_test.jpg', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
