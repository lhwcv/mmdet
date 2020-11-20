import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test']
classes = ["head"]


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id, root_dir):
    in_file = open(root_dir+'/Annotations_Quantize/%s.xml' % (image_id))
    #print(root_dir+'/Annotations/%s.xml' % (image_id))
    out_file = open(root_dir+'/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def main():
    root_dir = '/home/lhw/data/HollyWoodHead/'
    for set in sets:
        if not os.path.exists(root_dir+'/labels/'):
            os.makedirs(root_dir+'/labels/')
        image_ids = open(root_dir+'/ImageSets/Main/%s.txt' % (set)).read().strip().split()
        for image_id in image_ids:
            convert_annotation(image_id, root_dir)

if __name__ == '__main__':
    main()

