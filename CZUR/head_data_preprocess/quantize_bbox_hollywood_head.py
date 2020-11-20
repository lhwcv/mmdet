import  os
import xml.etree.ElementTree as ET

'''
<annotation>
        <folder>HollywoodHeads</folder>
        <filename>mov_002_147356.jpeg</filename>
        <source>
                <database>HollywoodHeads 2015 Database</database>
                <annotation>HollywoodHeads 2015</annotation>
                <image>WILLOW</image>
        </source>
        <size>
                <width>704</width>
                <height>416</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>head</name>
                <bndbox>
                        <xmin>407</xmin>
                        <ymin>11</ymin>
                        <xmax>693</xmax>
                        <ymax>352</ymax>
                </bndbox>
                <difficult>0</difficult>
        </object>
        <object>
                <name>head</name>
                <bndbox>
                        <xmin>38</xmin>
                        <ymin>41</ymin>
                        <xmax>302</xmax>
                        <ymax>378</ymax>
                </bndbox>
                <difficult>0</difficult>
        </object>
</annotation>
'''

from lxml import etree
class HollyHead_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "HollywoodHeads"
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        child3 = etree.SubElement(self.root, "source")
        child4 = etree.SubElement(child3, "annotation")
        child4.text = "HollywoodHeads 2015"
        child5 = etree.SubElement(child3, "database")
        child5.text = "HollywoodHeads 2015"
        child6 = etree.SubElement(child3, "image")
        child6.text = "WILLOW"


    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def add_obj(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        name = etree.SubElement(object, "name")
        name.text = label
        pose = etree.SubElement(object, "pose")
        pose.text = str(0)
        truncated = etree.SubElement(object, "truncated")
        truncated.text = str(0)
        difficult = etree.SubElement(object, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)

    def save_as_xml(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')


def  quantize_head_anno(
                   in_anno_file,
                   save_anno_file,
                   width_quantize_block = 50,
                   height_quantize_block = 50,
                   min_size_ratio = (0.08,  0.08)):
    tree = ET.parse(in_anno_file)
    root = tree.getroot()
    dst_ann = HollyHead_Annotations(root.find('filename').text)

    size = root.find('size')
    dw = int(size.find('width').text)
    dh = int(size.find('height').text)
    min_size = dw * dh * min_size_ratio[0] * min_size_ratio[1]
    dst_ann.set_size(dw, dh,3)
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)

        s = abs((ymax - ymin) * (xmax - xmin))
        if s > min_size:
            xmin = (xmin // width_quantize_block)
            xmin = width_quantize_block * xmin
            xmin = max(xmin, 0)

            xmax = (xmax // width_quantize_block) + 1.0
            xmax = width_quantize_block * xmax
            xmax = min(dw-1, xmax)

            ymin = (ymin // height_quantize_block)
            ymin = height_quantize_block * ymin
            ymin = max(ymin, 0)

            ymax = (ymax // height_quantize_block) + 1.0
            ymax = height_quantize_block * ymax
            ymax = min(dh, ymax)

            dst_ann.add_obj('head', xmin, ymin, xmax, ymax)

    dst_ann.save_as_xml(save_anno_file)