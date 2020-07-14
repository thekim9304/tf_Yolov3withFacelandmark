import os
from xml.etree.ElementTree import Element, SubElement, ElementTree


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def make_xml(folder, filename, filepath, image_shape, bboxs, landmarks):
    """
    :param savepath: A Save path that xml file is saved
    :param folder: A folder that placed an images
    :param filename: Image which correspond to annotation
    :param filepath: Full path that image is placed
    :param image_shape: tuple (h, w, c)
    :param bboxs: list, shape:(N, 4)
    :param landmarks: list(str), shape:(N, 196)
    :return:
    """
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = filepath

    size = SubElement(root, 'size')
    SubElement(size, 'height').text = str(image_shape[0])
    SubElement(size, 'width').text = str(image_shape[1])
    SubElement(size, 'depth').text = str(image_shape[2])

    for i, bbox in enumerate(bboxs):
        object = SubElement(root, 'object')
        bbox = SubElement(object, 'bbox')
        SubElement(bbox, 'xmin').text = str(bboxs[i][0])
        SubElement(bbox, 'ymin').text = str(bboxs[i][1])
        SubElement(bbox, 'xmax').text = str(bboxs[i][2])
        SubElement(bbox, 'ymax').text = str(bboxs[i][3])
        SubElement(object, 'landmarks').text = landmarks[i][0]

    indent(root)
    return ElementTree(root)

def main():
    make_xml('./', 'images', 'raccoon-1.jpg', '/Users/home/raccoon-1.jpg', (416, 416, 3),
             [[10, 10, 50, 50], [10, 10, 50, 50]])


if __name__ == '__main__':
    main()
