import numpy as np
import torch
import os
import json
import Preprocessing

from general_utils.environment_variables import get_dataset_dir

from Datasets.MuPoTS.RawDataset import MUPO3D_PATH
import ImageTools.ImageTools as ImageTools

import cv2

def load_samples(dataset_dir):
    ann_file = os.path.join(dataset_dir, 'ann.json')
    with open(ann_file, 'r') as f:
        samples = json.load(f)
    return samples

def load_intrinsics(dataset_dir):
    intrinsics_file = os.path.join(dataset_dir, 'intrinsics.json')
    with open(intrinsics_file, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics

class PreprocessedDataset():
    def __init__(self, dataset_dir=None, normalize_skeletons=False):
        if dataset_dir is None:
            dataset_dir = get_dataset_dir()
        dataset_dir = os.path.join(dataset_dir, MUPO3D_PATH)
        samples = load_samples(dataset_dir)
        intrinsics = load_intrinsics(dataset_dir)
        self.dataset_dir = dataset_dir
        self.samples = samples
        self.intrinsics = intrinsics
        self.output_size = 224
        self.normalize_skeletons = normalize_skeletons

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_data = sample['input']
        pos3d = np.array(sample['gt'])
        if self.normalize_skeletons:
            pos3d *= 1715/1660*1490/compute_height(pos3d) # ideal length / empirical length in PW3d training set between hands * empirical length between wrists in PW3D.
        recording = input_data[0]
        frame = input_data[1]
        intrinsic_camera = self.intrinsics[recording-1]
        full_image_path = os.path.join(self.dataset_dir, 'TS{}'.format(recording), 'img_{:06}.jpg'.format(frame))
        with open(full_image_path, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        im_shape = im.shape
        bbx = input_data[2:]
        bbx[0] = np.clip(bbx[0], 0, im_shape[1])
        bbx[1] = np.clip(bbx[1], 0, im_shape[1])
        bbx[2] = np.clip(bbx[2], 0, im_shape[0])
        bbx[3] = np.clip(bbx[3], 0, im_shape[0])
        w = bbx[1]-bbx[0]
        h = bbx[3]-bbx[2]
        size = max(w,h)
        downsample_factor = int(np.ceil(size/448))
        if downsample_factor != 1:
            start_x, end_x = get_bounds_in_range(0, im_shape[1], im_shape[1], downsample_factor)
            start_y, end_y = get_bounds_in_range(0, im_shape[0], im_shape[0], downsample_factor)
            bbx[0] = (bbx[0]-start_x)/downsample_factor
            bbx[1] = (bbx[1]-start_x)/downsample_factor
            bbx[2] = (bbx[2]-start_y)/downsample_factor
            bbx[3] = (bbx[3]-start_y)/downsample_factor
        else:
            start_x = 0
            end_x = im_shape[1]
            start_y = 0
            end_y = im_shape[0]
        top_left_x = (start_x+downsample_factor/2 -1/2)/downsample_factor # remove 1/2 to offset center of pixel
        top_left_y = (start_y+downsample_factor/2 -1/2)/downsample_factor # remove 1/2 to offset center of pixel
        intrinsic_downscale = np.array([[1/downsample_factor,0,-top_left_x],
                                        [0, 1/downsample_factor, -top_left_y],
                                        [0,0,1]])
        intrinsic_post_downscale = np.matmul(intrinsic_downscale, intrinsic_camera)
        if downsample_factor == 1:
            image_downsample = im.astype(np.float32)/255
        else:
            image_downsample = np.zeros(((end_y+1-start_y)//downsample_factor, (end_x+1-start_x)//downsample_factor, 3), dtype=np.uint16)
            for i in range(downsample_factor):
                for j in range(downsample_factor):
                    image_downsample += im[i+start_y:end_y+1:downsample_factor, j+start_x:end_x+1:downsample_factor]
            image_downsample = (image_downsample.astype(np.float32)/255)/(downsample_factor**2)

        w = (bbx[1]-bbx[0])*1.3
        h = (bbx[3]-bbx[2])*1.3
        mid_x = (bbx[0]+bbx[1])/2
        mid_y = (bbx[2]+bbx[3])/2
        bbx = [mid_x-w/2, mid_x+w/2, mid_y-h/2, mid_y+h/2]
        up_in_cam = np.array([0,1,0])
        r = Preprocessing.get_transform_to_ideal_camera(self.output_size, bbx, intrinsic_post_downscale, desired_up = up_in_cam)
        real_cam_to_ideal_cam, ideal_intrinsic, cam_im_to_ideal_im = r
        f = ideal_intrinsic[0,0] # focal length in pixels
        ideal_cam_to_real_cam = np.linalg.inv(real_cam_to_ideal_cam)
        image_mapping = ImageTools.NpAffineTransforms(cam_im_to_ideal_im)
        output_size = (self.output_size, self.output_size)
        image_ideal = ImageTools.np_warp_im_bilinear(image_downsample, image_mapping, output_size)

        annotated = np.zeros((28), dtype=np.bool)
        pos3d_all = np.zeros((28,3), dtype=np.float32)
        all_to_ann = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]
        annotated[all_to_ann] = 1
        pos3d_all[all_to_ann] = pos3d.transpose()
        R = real_cam_to_ideal_cam[:3,:3]
        pos3d_in_ideal = np.array(list(map(lambda p: np.matmul(R,p), pos3d_all))).astype(np.float32)
        return image_ideal.transpose(2,0,1), f, pos3d_in_ideal, annotated, True

def compute_height(gt):
    # 17x3
    path = [4,3,2,1,5,6,7]
    start = path[:-1]
    end = path[1:]
    height = np.sum(np.linalg.norm(gt[:,start]-gt[:,end], axis=0))
    return height

# copy from mpiiinf PreprocDataset
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
