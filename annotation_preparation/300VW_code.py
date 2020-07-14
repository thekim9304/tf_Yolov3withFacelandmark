import os
import cv2
import numpy as np
from annotation_preparation.make_xml import make_xml

data_info_path = 'E:/DB_FaceLandmark/300VW/info_data_split.txt'

db_path = 'E:/DB_FaceLandmark/300VW'

train_path = 'E:/DB_FaceLandmark/300VW_frame/train'
test_path = 'E:/DB_FaceLandmark/300VW_frame/test'

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)

save_path_train = './300VW_train2'
save_path_test = './300VW_test'

if not os.path.exists(save_path_train):
    os.mkdir(save_path_train)
if not os.path.exists(save_path_test):
    os.mkdir(save_path_test)
save_path_train = os.path.join(save_path_train, 'annotations')
if not os.path.exists(save_path_train):
    os.mkdir(save_path_train)
save_path_test = os.path.join(save_path_test, 'annotations')
if not os.path.exists(save_path_test):
    os.mkdir(save_path_test)

with open(data_info_path) as anno_info:
    lines = anno_info.read()

    train_num_list = list(map(int, lines.split('\n')[0].split('=')[-1].split(' ')))
    test1_num_list = list(map(int, lines.split('\n')[1].split('=')[-1].split(' ')))
    test2_num_list = list(map(int, lines.split('\n')[2].split('=')[-1].split(' ')))
    test3_num_list = list(map(int, lines.split('\n')[3].split('=')[-1].split(' ')))

anno_save_path = save_path_train
num_list = train_num_list
# anno_save_path = save_path_test
# num_list = test1_num_list + test2_num_list + test3_num_list

# 0 ~ 48
fishing_landmarks = np.arange(48)

for i in num_list:
    num_avi = '{:03d}'.format(i)

    annot_path = os.path.join(db_path, num_avi, 'annot')
    pts_paths = os.listdir(annot_path)

    avi_path = os.path.join(db_path, num_avi, 'vid.avi')
    print('Start frames extract on {}'.format(avi_path))

    cap = cv2.VideoCapture(avi_path)
    if cap.isOpened():
        frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('frames : {}, fps : {}, width : {}, height : {}'.format(frame_cnt, fps, width, height))

        frame_cnt = 0
        while True:
            ret, frame = cap.read()
            if ret:
                drawed_frame = frame.copy()
                '''
                Check landmarks code start
                '''
                x1, y1, x2, y2 = width, height, 0, 0

                file_path = os.path.join(annot_path, pts_paths[frame_cnt])

                landmarks_str = ''
                with open(file_path) as file:
                    texts = file.read().split('\n')
                    landmarks_text = texts[3:-2]

                    landmarks_text = np.array(landmarks_text)[fishing_landmarks]
                    landmarks_text = list(landmarks_text)

                    for landmark in landmarks_text:
                        x, y = landmark.split(' ')
                        landmarks_str += x
                        landmarks_str += (' ' + y + ' ')

                        x = int(float(x))
                        y = int(float(y))
                        # cv2.circle(drawed_frame, (x, y), 2, (255, 255, 255), -1)

                        if x < x1:
                            x1 = x
                        elif x > x2:
                            x2 = x
                        if y < y1:
                            y1 = y
                        elif y > y2:
                            y2 = y

                    # cv2.rectangle(drawed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                '''
                Check landmarks code end
                '''
                # cv2.imshow('video', frame)
                # cv2.imshow('draw_video', drawed_frame)

                if (frame_cnt % fps) == 0:
                    # check_write_path = os.path.join(train_path + '_check', '{}_{:04d}.jpg'.format(num_avi, frame_cnt))
                    # check_write_path = os.path.join(test_path + '_check', '{}_{:04d}.jpg'.format(num_avi, frame_cnt))
                    # cv2.imwrite(check_write_path, drawed_frame)
                    #
                    write_path = os.path.join(train_path, '{}_{:04d}.jpg'.format(num_avi, frame_cnt))
                    # write_path = os.path.join(test_path, '{}_{:04d}.jpg'.format(num_avi, frame_cnt))
                    # cv2.imwrite(write_path, frame)

                    # def make_xml(savepath, folder, filename, filepath, image_shape, bboxs, landmarks):
                    savepath = anno_save_path
                    folder = os.path.split(os.path.dirname(write_path))[-1]
                    filename = os.path.split(write_path)[-1]
                    filepath = write_path
                    filepath = os.path.join(filepath.split('/')[-2], filepath.split('/')[-1].split('\\')[0], filepath.split('/')[-1].split('\\')[1])
                    image_shape = frame.shape
                    bboxs = np.expand_dims([x1, y1, x2, y2], axis=0)
                    landmarks = np.expand_dims([landmarks_str[:-1]], axis=0)
                    tree = make_xml(folder, filename, filepath, image_shape, bboxs, landmarks)
                    tree.write(os.path.join(savepath, os.path.splitext(filename)[0]+'.xml'))

                frame_cnt += 1
                # if cv2.waitKey(int(fps)) == 27:
                if cv2.waitKey(1) == 27:
                    break
            else:
                break

        print('{}_avi was extracted {} frames!!'.format(i, frame_cnt))

    cap.release()
    cv2.destroyAllWindows()
