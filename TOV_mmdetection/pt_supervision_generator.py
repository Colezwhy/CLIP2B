#@TODO: to write a dataset transformer which transforms the VOC2007detection into point annotations
import numpy
import numpy as np
import torch
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import os, cv2


path = '/home/junweizhou/Datasets/VOC/VOC2012/Annotations'
save_path = '/home/junweizhou/Datasets/VOC/VOC2012/Annotations_qc_pt'
file_list = os.listdir(path)

for anno in file_list:
    xml_path = os.path.join(path, anno)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    file_name = root.find('filename').text
    file_number = file_name[0:-4]
    xml_name = file_number + '.xml'
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = int(float(object.find('bndbox').find('xmin').text))
        Ymin = int(float(object.find('bndbox').find('ymin').text))
        Xmax = int(float(object.find('bndbox').find('xmax').text))
        Ymax = int(float(object.find('bndbox').find('ymax').text))
        ori_pt_x = numpy.int64((Xmin + Xmax) // 2)
        noise_x = (Xmax - Xmin) // 15
        pt_x = int(numpy.random.normal(ori_pt_x, noise_x, 1))
        ori_pt_y = numpy.int64((Ymin + Ymax) // 2)
        noise_y = (Ymax - Ymin) // 15
        pt_y = int(numpy.random.normal(ori_pt_y, noise_y, 1))
        node = ET.SubElement(object, 'point')

        node_x = ET.SubElement(node, 'X')
        node_x.text = '%s' % pt_x
        node_y = ET.SubElement(node, 'Y')
        node_y.text = '%s' % pt_y


    tree.write(os.path.join(save_path, xml_name))




