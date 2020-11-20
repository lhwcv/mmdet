import  cv2
import  os
import  random
import xml.etree.ElementTree as ET

def main():
    data_root = '/home/lhw/data/HollyWoodHead/'
    anno_dir  = '/home/lhw/data/HollyWoodHead/Annotations_Quantize/'
    image_dir = '/home/lhw/data/HollyWoodHead/JPEGImages/'
    fns = os.listdir(anno_dir)
    fn = random.sample(fns, 1)[0]
    print('pick: ', fn)

    img = cv2.imread(image_dir+'/'+fn.replace('.xml', '.jpeg'))
    tree = ET.parse(anno_dir+'/'+fn)
    root = tree.getroot()
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      (0,0,255), 7,16)
    cv2.imwrite(data_root+'/random_pick.jpg', img)


if __name__ == '__main__':
    main()