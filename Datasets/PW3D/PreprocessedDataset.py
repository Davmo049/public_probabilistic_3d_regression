import os
import json
from Datasets.PW3D.RawDataset import DATASET_PREPROC_PATH

import numpy as np
from general_utils.environment_variables import get_dataset_dir
import Preprocessing
import ImageTools.ImageTools as ImageTools

import cv2

class Pw3dDataset():
    def __init__(self, samples, dataset_dir=None, output_size=224, augment=True, normalize_height=False):
        if dataset_dir is None:
            dataset_dir = get_dataset_dir()
        dataset_dir = os.path.join(dataset_dir, DATASET_PREPROC_PATH)
        self.samples = samples
        self.dataset_dir = dataset_dir
        self.output_size = output_size
        self.augment=augment
        self.normalize_height = normalize_height

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_idx = self.samples[idx]
        ann_path = os.path.join(self.dataset_dir, 'ann', '{}.json'.format(sample_idx))
        image_path = os.path.join(self.dataset_dir, 'images', '{}.jpg'.format(sample_idx))
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        intrinsic = np.array(ann['intrinsic'])
        pos_3d = np.array(ann['gt'])
        if self.normalize_height:
            height = compute_height(pos_3d)
            pos_3d *= 1715/height # average of male and female height
        bbx = ann['bbx']
        with open(image_path, 'rb') as f:
            buf = f.read()
        buf = np.frombuffer(buf, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        image = image[:,:,::-1].astype(np.float32)/255
        pos_3d_3dp = np.zeros((28,3))
        remap = [4, 18, 23, 3, 19, 24, 0, 20, 25, 1, 22, 27, 5, 8,13, 6, 9, 14, 10, 15, 11, 16, 12, 17]
        pos_3d_3dp[remap] = pos_3d.transpose()
        valid = np.zeros((28), dtype=np.bool)
        valid[remap] = 1
        R = np.eye(3)
        R[1,1] = -1
        calib_data = {'I': intrinsic, 'R': R}
        to_net_input = self.get_transform(bbx, calib_data, self.output_size, self.augment)
        image_mapping, real_cam_to_ideal_cam, ideal_intrinsic, flip = to_net_input
        if flip:
            pos3d_flipped = np.empty((28,3), dtype=np.float32)
            pos3d_flipped[:8] = pos_3d_3dp[:8] # spine/neck/head
            pos3d_flipped[8:13] = pos_3d_3dp[13:18] # arm/hand
            pos3d_flipped[13:18] = pos_3d_3dp[8:13] # arm/hand
            pos3d_flipped[18:23] = pos_3d_3dp[23:] # leg/foot
            pos3d_flipped[23:] = pos_3d_3dp[18:23] # leg/foot
            # valid does not need to flip, since valid is symmetric w.r.t. flips
            # e.g. left elbow -> right elbow, both valid
        else:
            pos3d_flipped = pos_3d_3dp

        R = real_cam_to_ideal_cam[:3,:3]
        pos3d_in_ideal_cam = np.array(list(map(lambda p: np.matmul(R, p), pos3d_flipped))).astype(np.float32)
        f = ideal_intrinsic[0,0] # focal length in pixels
        output_size = (self.output_size, self.output_size)
        image_ideal = ImageTools.np_warp_im_bilinear(image, image_mapping, output_size)
        return image_ideal.transpose(2,0,1), f, pos3d_in_ideal_cam.astype(np.float32), valid, True

    # from mpiinf
    @staticmethod
    def get_transform(bounding_box, calib_data, output_size, augment):
        translate_strength = 0.02
        rotation_strength = 0.1
        max_upscale = np.log(0.75)
        max_downscale = np.log(0.85)
        up_in_cam = -np.array(calib_data['R'])[:, 1] # up in world = [0,1,0], minus because image y axis is pointing down
        if augment:
            mid_x = (bounding_box[0] + bounding_box[2])/2
            mid_y = (bounding_box[2] + bounding_box[3])/2
            height = (bounding_box[2] - bounding_box[0])
            mid_x += translate_strength * np.random.uniform(-1,1) * height
            mid_y += translate_strength * np.random.uniform(-1,1) * height
            up_in_cam += rotation_strength * np.random.normal(size=(3))
            up_in_cam = up_in_cam/np.linalg.norm(up_in_cam)
        intrinsic = np.array(calib_data['I'])
        r = Preprocessing.get_transform_to_ideal_camera(output_size, bounding_box, intrinsic, desired_up=up_in_cam)
        real_cam_to_ideal_cam, ideal_intrinsic, cam_im_to_ideal_im = r
        transforms = []
        transforms.append(ImageTools.NpAffineTransforms(cam_im_to_ideal_im))
        if not augment:
            scale = np.exp((max_downscale+max_upscale)/2)
            scale = np.array([scale, scale])
            scale_center = np.array([1,1.0])*output_size/2
            transforms.append(ImageTools.scale_as_affine(0, scale, scale_center=scale_center))
            ideal_intrinsic = np.copy(ideal_intrinsic)
            ideal_intrinsic[0,0] *= scale[0]
            ideal_intrinsic[1,1] *= scale[0]
            flip = False
        else:
            # depth poorly defined for different scaling for different axis or projective
            scale = np.exp(np.random.uniform(max_downscale, max_upscale))
            scale = np.array([scale, scale])
            scale_center = np.array([1,1.0])*output_size/2
            transforms.append(ImageTools.scale_as_affine(0, scale, scale_center=scale_center))
            ideal_intrinsic = np.copy(ideal_intrinsic)
            ideal_intrinsic[0,0] *= scale[0]
            ideal_intrinsic[1,1] *= scale[0]
            # flip
            flip = bool(np.random.randint(2))
            if flip:
                transforms.append(ImageTools.fliplr_as_affine([output_size, output_size]))
                real_cam_to_ideal_cam[0] *= -1
        image_mapping = ImageTools.stack_affine_transforms(transforms)
        return image_mapping, real_cam_to_ideal_cam, ideal_intrinsic, flip



TRAIN_END = 22735
VAL_END = 32664
TEST_END = 68179
def get_normal(validate=True, normalize_train=False, normalize_val=False):
    samples_train = list(range(TRAIN_END))
    if validate:
        samples_val = list(range(TRAIN_END, VAL_END))
    else:
        samples_val = list(range(VAL_END, TEST_END)) # i.e. test
        # samples_val = list(range(VAL_END+415*10, VAL_END+417*10)) # example of incorrect height
    return Pw3dDataset(samples_train,normalize_height=normalize_train), Pw3dDataset(samples_val, augment=False, normalize_height=normalize_val)

def compute_height(gt):
    # gt is 3x17
    path = [23,21,19,17,14,12,13,16,18,20,22]
    start = path[:-1]
    end = path[1:]
    height = np.sum(np.linalg.norm(gt[:,end]-gt[:,start], axis=0))
    return height

def main():
    samples_train = list(range(TRAIN_END))
    ds = Pw3dDataset(samples_train)
    samples_height = samples_train[::10]
    h = []
    for i in samples_height:
        ann_path = os.path.join(ds.dataset_dir, 'ann', '{}.json'.format(i))
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        pos_3d = np.array(ann['gt'])
        height = compute_height(pos_3d)
        h.append(height)
    print(np.mean(h))

if __name__ == '__main__':
    main()
