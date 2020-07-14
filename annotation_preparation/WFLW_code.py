# https://wywu.github.io/projects/LAB/WFLW.html

# show landmarks
import os
import cv2
import numpy as np
from annotation_preparation.make_xml import make_xml

annotation_path = 'E:/DB_FaceLandmark/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train' \
                  '.txt '
image_path = 'E:/DB_FaceLandmark/WFLW/WFLW_images/'

save_path = './annotation_WFLW'

if not os.path.exists(save_path):
    os.mkdir(save_path)


fishing_landmarks = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 126, 127, 128, 129, 130, 131, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 146, 147, 150, 151]


with open(annotation_path) as anno_txts:
    data = anno_txts.read()

    annotations = data.split('\n')

    print(len(annotations))

    for i, anno_str in enumerate(annotations):
        anno_list = list(anno_str.split(' '))

        landmarks_anno = anno_list[:196]
        landmarks_anno = np.array(landmarks_anno)[fishing_landmarks]
        landmarks_anno = list(landmarks_anno)

        bbox_anno = anno_list[196:200]
        img_anno = anno_list[-1]

        img_path = str(image_path + img_anno)

        # def make_xml(savepath, folder, filename, filepath, image_shape, bboxs, landmarks):
        save_path = './annotation_WFLW'
        folder = img_anno.split('/')[0]
        filename = img_anno.split('/')[1]
        filepath = os.path.join(img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1])

        print(img_path)
        img = cv2.imread(img_path)
        image_shape = img.shape
        bboxs = np.expand_dims(bbox_anno, axis=0)
        # list to str
        landmarks = np.expand_dims([' '.join(landmarks_anno)], axis=0)

        tree = make_xml(folder, filename, filepath, image_shape, bboxs, landmarks)
        tree.write(os.path.join(save_path, str(i) + '_' + os.path.splitext(filename)[0] + '.xml'))
        print(i, end=' ')
        print('Saved {}'.format(os.path.join(save_path, os.path.splitext(filename)[0] + '.xml')))

    print('{} creat done!'.format(save_path))