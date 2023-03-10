import csv
import os
import hashlib
import requests
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm

import Datasets.OpenImages.find_reasonably_large_square as find_reasonably_large_square

from general_utils.environment_variables import get_dataset_dir

import cv2

openimages_lables_dirname = 'openimages_labels'

OPENIMAGES_MASKS = 'openimages_masks'
def download_masks(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    mask_dir = os.path.join(dataset_dir, OPENIMAGES_MASKS)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    tmpfile = os.path.join(mask_dir, 'tmpfile.zip')
    download_str = 'https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{}.zip'
    for i in ['a', 'b', 'c', 'd', 'e', 'f']: #[0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']:
        cur_url = download_str.format(i)
        r = requests.get(cur_url)
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        with open(tmpfile, 'wb') as f:
            f.write(r.content)
        r.close()
        shutil.unpack_archive(tmpfile, mask_dir)
        os.remove(tmpfile)

def create_plaintext_to_classidentifiers(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    filename = os.path.join(dataset_dir, openimages_lables_dirname, 'class-descriptions-boxable.csv')
    ret = {}
    with open(filename, 'r') as f:
        classident = csv.reader(f)
        for row in classident:
            ret[' '.join(row[1:]).lower()] = row[0]
    return ret

def load_class_datasets(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    filename = os.path.join(dataset_dir, openimages_lables_dirname, 'train-annotations-human-imagelabels-boxable.csv')
    ret = []
    with open(filename, 'r') as f:
        classident = csv.reader(f)
        seen_desc = False
        for row in classident:
            if not seen_desc:
                print(row)
                assert(row[0] == 'ImageID')
                assert(row[1] == 'Source')
                assert(row[2] == 'LabelName')
                assert(row[3] == 'Confidence')
                seen_desc = True
                continue
            ret.append((row[0], row[1], row[2], float(row[3])))
    return ret

def load_source_datasets(images_to_keep, dataset_dir=None):
    images_to_keep = set(images_to_keep)
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    filename = os.path.join(dataset_dir, openimages_lables_dirname, 'image_ids_and_rotation.csv')
    ret = {}
    with open(filename, 'r') as f:
        classident = csv.reader(f)
        seen_desc = False
        for row in classident:
            if not seen_desc:
                print(row)
                seen_desc = True
                continue
            if row[0] in images_to_keep:
                ret[row[0]] = row[1:]
    return ret

def main():
    x = create_plaintext_to_classidentifiers()
    xi = {v:k for k, v in x.items()}
    class_datasets = load_class_datasets()
    image_to_labels = {}
    for imid, source, labelname, confidence in class_datasets:
        if confidence == 1:
            classes = image_to_labels.get(imid, set())
            classes.add(labelname)
            image_to_labels[imid] = classes

    should_download_masks=False
    download_bg_images = False
    download_cloth_textures = False
    if should_download_masks:
        download_masks()
    if download_bg_images:
        max_bg_images = 2000
        bg_string_classes = ['lighthouse', 'loveseat', 'nail (construction)', 'oven', 'office building', 'printer', 'waste container', 'studio couch', 'taxi', 'washing machine']
        banned_bg_string_classes = ['human eye', 'human beard', 'person', 'boy', 'human mouth', 'human body', 'human foot', 'human leg', 'human ear', 'woman', 'human hair', 'human head', 'man', 'girl', 'human face', 'human arm', 'human nose', 'human hand']
 
        for a in bg_string_classes + banned_bg_string_classes:
            if a not in x:
                print(a)
        banned_bg_classes = list(map(lambda v: x[v], banned_bg_string_classes))
        bg_classes = list(map(lambda v: x[v], bg_string_classes))
        bg_images = []
        for image, classes in image_to_labels.items():
            banned_found = False
            for cneg in banned_bg_classes: 
                if cneg in classes:
                    banned_found = True
                    break
            if banned_found:
                continue
            for cpos in bg_classes: 
                if cpos in classes:
                    bg_images.append(image)
                    break
        np.random.seed(12345)
        bg_images_source = load_source_datasets(bg_images)
        bg_images_source = [[k] + v for k,v in bg_images.items()]
        np.random.shuffle(bg_images_source)
        bg_images_source = bg_images_source[:max_bg_images]
        download_bg_dataset(bg_images_source)
    if download_cloth_textures:
        max_cloth_textures = 10000
        cloth_string_classes = ['backpack', 'clothing', 'dress', 'jeans', 'miniskirt', 'scarf', 'trousers']
        for a in cloth_string_classes:
            if a not in x:
                print(a)
        cloth_classes = list(map(lambda v: x[v], cloth_string_classes))
        cloth_images = []
        for image, classes in image_to_labels.items():
            for cpos in cloth_classes: 
                if cpos in classes:
                    cloth_images.append(image)
                    break
 
        np.random.seed(12345)
        cloth_images_with_boxes = load_cloth_texture_boxes(cloth_images, cloth_classes, max_cloth_textures)
        imageids = list(cloth_images_with_boxes.keys())
        source_imageids = load_source_datasets(imageids)
        cloth_images_source_with_boxes = []
        for imageid, boxdata in cloth_images_with_boxes.items():
            cloth_images_source_with_boxes.append((source_imageids[imageid], boxdata))
        np.random.shuffle(cloth_images_source_with_boxes)
        cloth_images_source_with_boxes = cloth_images_source_with_boxes[:max_cloth_textures]
        download_cloth_datasets(cloth_images_source_with_boxes)


OPENIMAGEDUMP = 'OpenImageDump'
def download_bg_dataset(source):
    dataset_dir = get_dataset_dir()
    openimage_data = os.path.join(dataset_dir, OPENIMAGEDUMP, 'bg')
    if os.path.exists(openimage_data):
        shutil.rmtree(openimage_data)
    os.makedirs(openimage_data)
    for s in source:
        imageid, subset, original_url, original_landing_url, licence, _,_,_,_,md5b64, _, rotation = s
        md5 = b64ToNum(md5b64)
        r = requests.get(original_url)
        content = r.content
        md5_recieved = hashlib.md5(content).hexdigest()
        expected_type = original_url.split('.')[-1]
        if md5_recieved == md5:
            with open(os.path.join(openimage_data, '{}.{}'.format(imageid, expected_type)), 'wb') as f:
                f.write(content)
        else:
            print('fail')
            print(md5)
            print(md5_recieved)
            print(imageid)
            print(original_url)
            print(original_landing_url)
            print(md5b64)
        r.close()

def b64ToNum(b64):
    # not super efficient, but good enough
    i = 0
    for d in bytes(b64, 'UTF8'):
        if 64 < d < 91:
            c = d-65
        elif 96 < d < 123:
            c = d-71
        elif 47 < d < 58:
            c = d+4
        elif d == 43:
            c = 62
        elif d == 47:
            c = 63
        else:
            continue
        i = i * 64+c
    i = i // 16
    r= hex(i)[2:]
    return '0'*(32-len(r))+r

def load_cloth_texture_boxes(images, classes, max_boxes, dataset_dir=None):
    images = set(images)
    classes = set(classes)
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    with open(os.path.join(dataset_dir, openimages_lables_dirname, 'train-annotations-object-segmentation.csv'), 'r') as f:
        loaded_boxes = {}
        for l in f.readlines():
            fields = l.strip().split(',')
            imagepath = fields[0]
            image_id = fields[1]
            classlabel = fields[2]
            if classlabel in classes:
                assert(image_id in images)
                current_boxes = loaded_boxes.get(image_id, [])
                mask_path = os.path.join(dataset_dir, OPENIMAGES_MASKS, imagepath)
                PIL_image = Image.open(mask_path)
                PIL_image.convert('RGB')
                image = np.array(PIL_image.getdata()).reshape(PIL_image.size[1], PIL_image.size[0])
                bbx = find_reasonably_large_square.find_reasonably_large_square(image)
                if bbx is None:
                    continue
                xmin, xmax, ymin, ymax = bbx
                if xmax-xmin < 50 or ymax-ymin < 50:
                    continue
                current_boxes.append((bbx, PIL_image.size))
                loaded_boxes[image_id] = current_boxes
                print('{}/{}'.format(len(loaded_boxes), max_boxes))
                if len(loaded_boxes) >= max_boxes:
                    return loaded_boxes


def download_cloth_datasets(cloth_images_source_with_boxes):
    dataset_dir = get_dataset_dir()
    openimage_data = os.path.join(dataset_dir, OPENIMAGEDUMP, 'cloth')

    if os.path.exists(openimage_data):
        shutil.rmtree(openimage_data)
    os.makedirs(openimage_data)
    num_saved = 0
    for imsource, boxdata in cloth_images_source_with_boxes:
        subset, original_url, original_landing_url, licence, _,_,_,_,md5b64, _, rotation = imsource
        md5 = b64ToNum(md5b64)
        r = requests.get(original_url)
        content = r.content
        md5_recieved = hashlib.md5(content).hexdigest()
        if md5_recieved == md5:
            im = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            im = im[:,:,::-1]
            for b in boxdata:
                xmin, xmax, ymin, ymax = b[0]
                scaley = im.shape[0]/b[1][1]
                scalex = im.shape[1]/b[1][0]
                xmin *= scalex
                xmax *= scalex
                ymin *= scaley
                ymax *= scaley
                xmin = int(np.round(xmin))
                xmax = int(np.round(xmax))
                ymin = int(np.round(ymin))
                ymax = int(np.round(ymax))
                cutout = im[ymin:ymax, xmin:xmax]
                outfile = os.path.join(openimage_data, '{}.png'.format(num_saved))
                PIL_im = Image.fromarray(cutout).convert('RGB')
                PIL_im.save(outfile)
                num_saved+=1
        else:
            print('fail')
            print(md5)
            print(md5_recieved)
            print(original_url)
            print(original_landing_url)
        r.close()



if __name__ == '__main__':
    main()
