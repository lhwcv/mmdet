import  os
import  argparse
import  glob
import tqdm
import xml.etree.ElementTree as ET
from  CZUR.head_data_preprocess.quantize_bbox_hollywood_head import  quantize_head_anno

CLASS_NAMES=[
    'head'
]

#MIN_SIZE_RATIO = [1/10.0 , 1/ 10.0]
MIN_SIZE_RATIO = [0.2 , 0.2]
MAX_HEAD_N = 4

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default='/home/lhw/m2_disk/data/HollyWoodHead/')
    p.add_argument("--train_file_save", type=str,
                   default='/home/lhw/m2_disk/data/HollyWoodHead/train.txt')
    p.add_argument("--val_file_save", type=str,
                   default='/home/lhw/m2_disk/data/HollyWoodHead/val.txt')
    p.add_argument("--train_ratio", type=float,
                   default=0.7)

    p.add_argument("--quantize_label", type=bool,
                   default=True)

    return  p.parse_args()

def  get_movie_frame_names_dict(img_data_dir, anno_dir):
    '''

    :param img_data_dir: JPEGImages
    :param anno_dir: Annotations
    :return:  {
                   'mov_001': [file1.jpg, file2.jpg .....],
                   'mov_002': [file1.jpg, file2.jpg .....]
                }
    '''
    data_dict = {}
    filenames = [f for f in os.listdir(img_data_dir) if 'mov' in f]
    print('hollywood head frames total: ', len(filenames))
    valid_n = 0

    for fn in tqdm.tqdm(filenames):
        xml_name = fn.replace('.jpeg','.xml')
        ann_fn = os.path.join(anno_dir,xml_name )
        if os.path.exists(ann_fn):
            has_obj = False
            tree = ET.parse(ann_fn)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name')
                if name is None:
                    break
                name = name.text
                if name in CLASS_NAMES:
                    has_obj = True
                    break
            if has_obj:
                ### check, if all object is small, break
                size = root.find('size')
                dw = int(size.find('width').text)
                dh = int(size.find('height').text)
                min_size = dw * dh * MIN_SIZE_RATIO[0] * MIN_SIZE_RATIO[1]

                size_valid = False
                n_obj = 0
                for obj in root.iter('object'):
                    n_obj += 1
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                             float(xmlbox.find('ymin').text),
                             float(xmlbox.find('ymax').text))
                    w = abs(b[1] - b[0])
                    h = abs(b[3] - b[2])
                    s = w * h
                    if s > min_size:
                        size_valid = True
                        break
                if size_valid and n_obj<=MAX_HEAD_N :
                    valid_n+=1
                    movie_id = xml_name.split('_')[1]
                    fn = os.path.join(img_data_dir,fn)
                    if movie_id not in data_dict.keys():
                        data_dict[movie_id] = [fn]
                    else:
                        data_dict[movie_id].append(fn)

    print('valid frames: ', valid_n)
    print('movies: ', len(data_dict.keys()) )
    return data_dict

def main():
    args = get_args()
    data_dir = args.data_dir
    train_file_save = args.train_file_save
    val_file_save   = args.val_file_save
    train_ratio = args.train_ratio

    print('min size: ', MIN_SIZE_RATIO)

    movie_frame_names_dict = get_movie_frame_names_dict(
        data_dir + '/JPEGImages',
        data_dir + '/Annotations',
    )

    movie_names = list(movie_frame_names_dict.keys() )
    N = int(len(movie_names) * train_ratio)
    train_movies = movie_names[:N]
    val_movies   = movie_names[N:]

    train_files = []
    for n  in train_movies:
        train_files.extend(movie_frame_names_dict[n])
    val_files = []
    for n in val_movies:
        val_files.extend(movie_frame_names_dict[n])
    print('train files n: ', len(train_files))
    print('val   files n: ', len(val_files))

    with open(train_file_save,'w') as f:
        for fn in train_files:
            f.write(fn+'\n')
    with open(val_file_save,'w') as f:
        for fn in val_files:
            f.write(fn+'\n')


    if args.quantize_label:
        quantize_anno_save_dir = data_dir + '/Annotations_Quantize/'
        if not os.path.exists(quantize_anno_save_dir):
            os.mkdir(quantize_anno_save_dir)
        for fn in train_files:
            fn = fn.split('/')[-1]
            fn = fn.replace('.jpeg', '.xml')
            anno_in = data_dir + '/Annotations/'+ fn
            anno_save = quantize_anno_save_dir+'/'+fn
            quantize_head_anno(anno_in, anno_save)
        for fn in val_files:
            fn = fn.split('/')[-1]
            fn = fn.replace('.jpeg', '.xml')
            anno_in = data_dir + '/Annotations/'+ fn
            anno_save = quantize_anno_save_dir+'/'+fn
            quantize_head_anno(anno_in, anno_save)

        ###write to ImageSets
        with open(data_dir+'/ImageSets/Main/trainval.txt', 'w') as f:
            for fn in train_files:
                fn = fn.split('/')[-1].split('.')[0]
                f.write(fn + '\n')
        with open(data_dir+'/ImageSets/Main/train.txt', 'w') as f:
            for fn in train_files:
                fn = fn.split('/')[-1].split('.')[0]
                f.write(fn + '\n')
        with open(data_dir+'/ImageSets/Main/val.txt', 'w') as f:
            for fn in val_files:
                fn = fn.split('/')[-1].split('.')[0]
                f.write(fn + '\n')
        with open(data_dir+'/ImageSets/Main/test.txt', 'w') as f:
            for fn in val_files:
                fn = fn.split('/')[-1].split('.')[0]
                f.write(fn + '\n')




if __name__ == '__main__':
    main()