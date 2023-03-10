import os
import json

import numpy as np
from PIL import Image

from general_utils.environment_variables import get_dataset_dir

import cv2

# uses github.com/mks0601/3DMPPE_ROOTNET_RELEASE annotations
# original labels were .pkl, which can encode arbitrary code exectution.

DATASET_PATH = '3DPW'
DATASET_PREPROC_PATH = '3DPW_Preproc'
ANN_DIR = 'sequenceFiles'
ANN_FILE_TRAIN = '3DPW_train.json'
ANN_FILE_VAL = '3DPW_validation.json'
ANN_FILE_TEST = '3DPW_test.json'

def main():
    dataset_dir = os.path.join(get_dataset_dir(), DATASET_PATH)
    image_dir = os.path.join(dataset_dir, 'imageFiles')
    preproc_dir = os.path.join(get_dataset_dir(), DATASET_PREPROC_PATH)
    preproc_image_dir = os.path.join(preproc_dir, 'images')
    preproc_ann_dir = os.path.join(preproc_dir, 'ann')
    if not os.path.exists(preproc_dir):
        os.makedirs(preproc_dir)
    if not os.path.exists(preproc_image_dir):
        os.makedirs(preproc_image_dir)
    if not os.path.exists(preproc_ann_dir):
        os.makedirs(preproc_ann_dir)
    # train
    ann_file = os.path.join(dataset_dir, ANN_DIR, ANN_FILE_TRAIN)
    with open(ann_file, 'r') as f:
        anns = json.load(f)
    annotations_train = anns['annotations']
    all_image_data = anns['images']
    # val
    ann_file = os.path.join(dataset_dir, ANN_DIR, ANN_FILE_VAL)
    with open(ann_file, 'r') as f:
        anns = json.load(f)
    annotations_val = anns['annotations']
    all_image_data += anns['images']

    # test
    ann_file = os.path.join(dataset_dir, ANN_DIR, ANN_FILE_TEST)
    with open(ann_file, 'r') as f:
        anns = json.load(f)
    annotations_test = anns['annotations']
    all_image_data += anns['images']

    save_idx = 0
    save_idx = preprocess_samples(all_image_data, annotations_train, save_idx, image_dir, preproc_image_dir, preproc_ann_dir)
    print(save_idx) #22735
    save_idx = preprocess_samples(all_image_data, annotations_val, save_idx, image_dir, preproc_image_dir, preproc_ann_dir)
    print(save_idx) #32664
    save_idx = preprocess_samples(all_image_data, annotations_test, save_idx, image_dir, preproc_image_dir, preproc_ann_dir)
    print(save_idx) #68179


def preprocess_samples(all_image_data, annotations, save_idx, image_dir, preproc_image_dir, preproc_ann_dir):
    for load_idx in range(0, len(annotations)):
        ann = annotations[load_idx]
        im_id = ann['image_id']
        im_data = all_image_data[im_id]
        assert(im_data['id'] == im_id)
        preprocess_sample(ann,im_data,save_idx, image_dir, preproc_image_dir, preproc_ann_dir)
        save_idx += 1
    return save_idx

def preprocess_sample(ann, im_data, save_idx, image_dir, preproc_image_dir, preproc_ann_dir):
    intrinsic = np.eye(3)
    intrinsic[0,0] = im_data['cam_param']['focal'][0]
    intrinsic[1,1] = im_data['cam_param']['focal'][1]
    intrinsic[0,2] = im_data['cam_param']['princpt'][0]
    intrinsic[1,2] = im_data['cam_param']['princpt'][1]
    file_name = 'image_{:05}.jpg'.format(im_data['frame_idx'])
    image_path = os.path.join(image_dir, im_data['sequence'], file_name)
    with open(image_path, 'rb') as f:
        data = f.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    joint_cam = np.array(ann['joint_cam']).transpose()
    joint_cam *= 1000
    joint_homo = joint_cam/joint_cam[2].reshape(1,-1)
    joint_2d = np.matmul(intrinsic, joint_homo)[:2]
    im = im[:,:,::-1]
    bbx_min_x = np.min(joint_2d[0])
    bbx_max_x = np.max(joint_2d[0])
    bbx_min_y = np.min(joint_2d[1])
    bbx_max_y = np.max(joint_2d[1])
    bbx = [bbx_min_x, bbx_max_x, bbx_min_y, bbx_max_y]
    w = bbx_max_x-bbx_min_x
    h = bbx_max_y-bbx_min_y
    size = max(w,h)
    size_save = size*1.4
    mid_x = (bbx_max_x+bbx_min_x)/2
    mid_y = (bbx_max_y+bbx_min_y)/2
    downsampling_rate = int(np.ceil(size//448))
    save_x_start = mid_x - size_save/2
    save_x_end = mid_x + size_save/2
    save_y_start = mid_y - size_save/2
    save_y_end = mid_y + size_save/2
    bbx_save = [save_x_start, save_x_end, save_y_start, save_y_end]

    bbx_ds, ds_r = get_downsample_bbx(bbx_save, im.shape)
    bbx_save = bbx_ds
    assert(ds_r < 16)
    image = im[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1],:]
    if ds_r != 1:
        downsampled_image = np.copy(image[::ds_r, ::ds_r]).astype(np.uint16)
        for i in range(ds_r):
            for j in range(ds_r):
                if i == 0 and j == 0:
                    continue
                downsampled_image += image[i::ds_r, j::ds_r,:]
        image = ((downsampled_image + ds_r**2/2) / ds_r**2).astype(np.uint8)
    intrinsic_downsample = np.array([[1/ds_r, 0, (-bbx_ds[0]+1/2)/ds_r-1/2], [0, 1/ds_r, (-bbx_ds[2]+0.5)/ds_r-0.5],[0,0,1]])
    intrinsic = np.matmul(intrinsic_downsample, intrinsic)
    bbx = [(bbx[0]-bbx_ds[0])/ds_r, (bbx[1]-bbx_ds[0])/ds_r, (bbx[2]-bbx_ds[2])/ds_r, (bbx[3]-bbx_ds[2])/ds_r]


    save_ann_path = os.path.join(preproc_ann_dir, '{}.json'.format(save_idx))
    frame_data = {'bbx': bbx, 'intrinsic': list(map(list, intrinsic)), 'gt': list(map(list, joint_cam))}
    with open(save_ann_path, 'w') as f:
        json.dump(frame_data, f)
    save_image_path = os.path.join(preproc_image_dir, '{}.jpg'.format(save_idx))
    im_save = Image.fromarray(image)
    im_save.save(save_image_path)


# copy from PreprocDataset
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
    downsample_rate = max(1, int(np.ceil(h/448)))
    if downsample_rate == 1:
        start_x = np.clip(start_x, 0, image_shape[1]-1)
        end_x = np.clip(end_x, 1, image_shape[1])
        start_y = np.clip(start_y, 0, image_shape[0]-1)
        end_y = np.clip(end_y, 1, image_shape[0])
    else:
        start_x, end_x = get_bounds_in_range(start_x, end_x, image_shape[1], downsample_rate)
        start_y, end_y = get_bounds_in_range(start_y, end_y, image_shape[0], downsample_rate)
    return [start_x, end_x, start_y, end_y], downsample_rate

# copy from PreprocDataset
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


if __name__ == '__main__':
    main()
