import os
import json
import subprocess
import shutil

import numpy as np
from PIL import Image
import tqdm

from general_utils.environment_variables import get_dataset_dir

import cv2

COCO_DIR_NAME = 'COCO'
COCO_PREPROC_DIR_NAME = 'CocoPreproc'

# this preprocessing "compresses" by discarding mostly unused information
# the dataset goes from ~21G->4G, also makes loading times more uniform and reduces frequency aliasing from downsampling
def main():
    dataset_dir = os.path.join(get_dataset_dir(), COCO_DIR_NAME)
    preproc_dir = os.path.join(get_dataset_dir(), COCO_PREPROC_DIR_NAME)
    os.makedirs(preproc_dir)
    os.makedirs(os.path.join(preproc_dir, 'images'))
    os.makedirs(os.path.join(preproc_dir, 'anns'))

    save_idx = 0
    # train
    annotation_path = os.path.join(dataset_dir, 'annotations', 'person_keypoints_train2017.json')
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    image_dir = os.path.join(dataset_dir, 'train2017')
    save_idx = preproc(annotations, image_dir, preproc_dir, save_idx)
    print(save_idx)
    # val
    annotation_path = os.path.join(dataset_dir, 'annotations', 'person_keypoints_val2017.json')
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    image_dir = os.path.join(dataset_dir, 'val2017')
    save_idx = preproc(annotations, image_dir, preproc_dir, save_idx)
    # save_idx = 101000
    print(save_idx)

def preproc(annotations, image_dir, preproc_dir, save_idx):
    image_id_to_annotation = {}
    for imdata in annotations['images']:
        image_id_to_annotation[int(imdata['id'])] = imdata
    for ann in tqdm.tqdm(annotations['annotations']):
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        shoulder_annotated = (keypoints[5,2] == 2 or keypoints[6,2] == 2)
        hip_annotated = (keypoints[11,2] == 2 or keypoints[12,2] == 2)
        if not(shoulder_annotated and hip_annotated):
            # remove some stupid samples
            continue
        im_data = image_id_to_annotation[int(ann['image_id'])]

        image_path = os.path.join(image_dir, im_data['file_name'])
        with open(image_path, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        xs = keypoints[:,0]
        ys = keypoints[:,1]
        valid = keypoints[:,2] != 0
        x_max = np.max(xs[valid])
        x_min = np.min(xs[valid])
        y_max = np.max(ys[valid])
        y_min = np.min(ys[valid])
        bbx = [x_min, x_max, y_min, y_max]
        bbx_ds, ds_r = get_downsample_bbx(bbx, im.shape)
        image_downsample_path = os.path.join(preproc_dir, 'images', '{}.jpg'.format(save_idx))
        save_ann_path = os.path.join(preproc_dir, 'anns', '{}.json'.format(save_idx))
        save_idx += 1
        if (ds_r == 1) and (bbx_ds[0] == 0) and (bbx_ds[2] == 0) and (bbx_ds[1] == im.shape[1]) and (bbx_ds[3] == im.shape[0]):
            shutil.copyfile(image_path, image_downsample_path)
        else:
            if ds_r == 1:
                w = bbx_ds[1]-bbx_ds[0]
                h = bbx_ds[3]-bbx_ds[2]
                crop_jpeg(image_path, image_downsample_path, w,h,bbx_ds[0],bbx_ds[2])
            else:
                image = im[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1],:]
                downsampled_image = np.copy(image[::ds_r, ::ds_r]).astype(np.uint16)
                for i in range(ds_r):
                    for j in range(ds_r):
                        if i == 0 and j == 0:
                            continue
                        downsampled_image += image[i::ds_r, j::ds_r,:]
                image = ((downsampled_image + ds_r**2/2) / ds_r**2).astype(np.uint8)
                im_save = Image.fromarray(image)
                im_save.save(image_downsample_path)
        pose = keypoints[:, :2].astype(np.float32)
        pose[:, 0] = (pose[:, 0] - bbx_ds[0] - (ds_r - 1)/2) / ds_r
        pose[:, 1] = (pose[:, 1] - bbx_ds[2] - (ds_r - 1)/2) / ds_r
        pose[np.logical_not(valid)] = 0
        bbx_adjusted = np.zeros((4))
        bbx_adjusted[0] = (bbx[0]-bbx_ds[0])/ds_r
        bbx_adjusted[1] = (bbx[1]-bbx_ds[0])/ds_r
        bbx_adjusted[2] = (bbx[2]-bbx_ds[2])/ds_r
        bbx_adjusted[3] = (bbx[3]-bbx_ds[2])/ds_r
        valid = keypoints[:,2]
        ann_save = {'pose': list(map(lambda x: list(map(float, x)), pose)), 'valid': list(map(int, valid)), 'bbx': list(map(float, bbx_adjusted))}
        with open(save_ann_path, 'w') as f:
            json.dump(ann_save, f)
    return save_idx



# copy from mpiinf PreprocDataset
def get_downsample_bbx(bbx, image_shape):
    start_x = int(bbx[0])
    end_x = int(bbx[1])
    start_y = int(bbx[2])
    end_y = int(bbx[3])
    w = (end_x-start_x)*1.3
    h = (end_y-start_y)*1.3
    h = max(w,h)
    mid_x = (start_x+end_x)/2
    mid_y = (start_y+end_y)/2
    start_x = int(np.round(mid_x-h/2))
    end_x = int(np.round(mid_x+h/2))
    start_y = int(np.round(mid_y-h/2))
    end_y = int(np.round(mid_y+h/2))
    downsample_rate = max(1, int(np.floor(h/224)))
    if downsample_rate == 1:
        start_x, end_x, start_y, end_y = get_perfect_jpeg_crop([start_x, end_x, start_y, end_y])
        start_x = np.clip(start_x, 0, image_shape[1]-1)
        end_x = np.clip(end_x, 1, image_shape[1])
        start_y = np.clip(start_y, 0, image_shape[0]-1)
        end_y = np.clip(end_y, 1, image_shape[0])
    else:
        start_x, end_x = get_bounds_in_range(start_x, end_x, image_shape[1], downsample_rate)
        start_y, end_y = get_bounds_in_range(start_y, end_y, image_shape[0], downsample_rate)
    return [start_x, end_x, start_y, end_y], downsample_rate

# copy from mpiinf PreprocDataset
def get_bounds_in_range(s, e, l, ds):
    assert(ds < l)
    s_i = (s // ds)*ds
    e_i = ((e // ds)+1)*ds
    if e_i > l and s_i < 0:
        w_final = (l // ds)*ds
        s_i = (l // 2)-w_final//2
        e_i = s_i + w_final
    elif e_i > l:
        e_i = l
        s_i = e_i - (((e_i-s)//ds)+1)*ds
        if s_i < 0:
            s_i += ds
    elif s_i < 0:
        s_i = 0
        e_i = ((e//ds)+1)*ds
        if e_i > l:
            e_i -= ds
    return s_i, e_i

def get_perfect_jpeg_crop(bbx):
    ret = []
    ret.append((bbx[0]//8)*8)
    ret.append(int(np.ceil(bbx[1]/8)*8))
    ret.append((bbx[2]//8)*8)
    ret.append(int(np.ceil(bbx[3]/8)*8))
    return ret

def crop_jpeg(path_src, path_dst, w,h,x,y):
    call = ['jpegtran', '-perfect', '-crop', '{}x{}+{}+{}'.format(w, h,x,y), '-outfile', path_dst, path_src]
    proc = subprocess.Popen(' '.join(call), shell=True)
    proc.wait()


if __name__ == '__main__':
    main()
