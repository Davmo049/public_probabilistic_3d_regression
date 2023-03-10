import os
import json

import scipy.io
import numpy as np
from PIL import Image
import tqdm

from general_utils.environment_variables import get_dataset_dir
import cv2

RAW_DATASET_DIR = 'mpii_pose_raw'
PREPROC_DATASET_DIR = 'mpii_pose_preprocessed'
PREPROC_ANN_DIR = 'ann'
ANNOTATION_DIR = 'mpii_human_pose_v1_u12_2'
ANNOTATION_FILE = 'mpii_human_pose_v1_u12_1.mat'
IMAGE_DIR = 'images'

def parse_pose_from_rect(rect):
    if 'annopoints' not in rect.dtype.names:
        return None
    annopoints = rect['annopoints']
    if len(annopoints) == 0:
        return None
    points = annopoints[0,0]['point'].reshape(-1)
    xs = list(map(lambda i: points[i]['x'][0,0], range(len(points))))
    ys = list(map(lambda i: points[i]['y'][0,0], range(len(points))))
    ids = list(map(lambda i: points[i]['id'][0,0], range(len(points))))
    annotated = np.zeros((16), dtype=bool)
    pos = np.zeros((16,2), dtype=np.float32)
    for x,y,i in zip(xs, ys, ids):
        pos[i,0] = x
        pos[i,1] = y
        annotated[i] = True
    return pos, annotated

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

def export_dataset(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    annotation_file = os.path.join(dataset_dir, RAW_DATASET_DIR, ANNOTATION_DIR, ANNOTATION_FILE)
    image_dir = os.path.join(dataset_dir, RAW_DATASET_DIR, IMAGE_DIR)
    save_dir = os.path.join(dataset_dir, PREPROC_DATASET_DIR)
    save_im_dir = os.path.join(save_dir, IMAGE_DIR)
    save_ann_dir = os.path.join(save_dir, PREPROC_ANN_DIR)
    os.makedirs(save_dir)
    os.makedirs(save_im_dir)
    os.makedirs(save_ann_dir)
    annotations = scipy.io.loadmat(annotation_file)
    annotations = annotations['RELEASE']
    single_person = annotations['single_person'][0,0].reshape(-1)
    image_train = annotations['img_train'][0,0].reshape(-1)
    annolist = annotations['annolist'][0,0].reshape(-1)
    num_samples = 0
    iters = -1
    for train, anno, sps in tqdm.tqdm(zip(image_train, annolist,single_person), total=len(image_train)):
        iters += 1
        im_name = anno['image'][0,0]['name'][0]
        impath = os.path.join(image_dir, im_name)
        if not(os.path.exists(impath)):
            print('skipping {}'.format(impath))
            continue
        with open(impath, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        if train:
            annorect = anno['annorect'].reshape(-1)
            for i, rect in enumerate(annorect):
                is_single = (i+1) in sps
                r = parse_pose_from_rect(rect)
                if r is None:
                    print('this index has no annopoints is missing {}, {}'.format(iters, i))
                    continue
                pose, is_annotated = r

                start_x = np.clip(np.min(pose[is_annotated,0]), 0, im.shape[1])
                end_x = np.clip(np.max(pose[is_annotated,0]), 0, im.shape[1])
                start_y = np.clip(np.min(pose[is_annotated,1]), 0, im.shape[0])
                end_y = np.clip(np.max(pose[is_annotated,1]), 0, im.shape[0])

                bbx = [start_x, end_x, start_y, end_y]

                bbx_ds, ds_r = get_downsample_bbx(bbx, im.shape)
                assert(ds_r < 16)
                if ds_r == 1:
                    image = im[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1],:]
                else:
                    image = im[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1],:]
                    downsampled_image = np.copy(image[::ds_r, ::ds_r]).astype(np.uint16)
                    for i in range(ds_r):
                        for j in range(ds_r):
                            if i == 0 and j == 0:
                                continue
                            downsampled_image += image[i::ds_r, j::ds_r,:]
                    image = ((downsampled_image + ds_r**2/2) / ds_r**2).astype(np.uint8)

                pose[is_annotated,0] -= bbx_ds[0] - 1/2 # 1/2 because I assume annotations refer to center of pixel, basis change to "top left coordinate system"
                pose[is_annotated,1] -= bbx_ds[2] - 1/2
                pose /= ds_r
                pose -= 1/2 # back to center of pixel coordinate system

                start_x -= bbx_ds[0] - 1/2
                end_x -= bbx_ds[0] - 1/2
                start_y -= bbx_ds[2] - 1/2
                end_y -= bbx_ds[2] - 1/2
                start_x /= ds_r
                start_y /= ds_r
                end_x /= ds_r
                end_y /= ds_r
                start_x -= 1/2
                start_y -= 1/2
                end_x -= 1/2
                end_y -= 1/2

                ann = {'single': is_single,
                        'pose': list(map(lambda x: list(map(float, x)), pose)),
                       'bbx': [start_x, end_x, start_y, end_y],
                       'annotated': list(map(int, is_annotated))
                      }
                ann_path = os.path.join(save_ann_dir, '{}.json'.format(num_samples))
                try:
                    im_save_path = os.path.join(save_im_dir, '{}.jpg'.format(num_samples))
                    im_save = Image.fromarray(image)
                    im_save.save(im_save_path)
                    with open(ann_path, 'w') as f:
                        json.dump(ann, f)
                    num_samples += 1
                except ValueError as e:
                    pass # tile cannot extend outside image when saving image

def main():
    export_dataset()

if __name__ == '__main__':
    main()


