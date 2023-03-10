import os
import json
import numpy as np
from general_utils.environment_variables import get_dataset_dir
import matplotlib.pyplot as plt
import matplotlib
from Datasets.mpii_pose.RawDataset import IMAGE_DIR, PREPROC_DATASET_DIR, PREPROC_ANN_DIR
import ImageTools.ImageTools as ImageTools

import cv2

class Dataset():
    def __init__(self,dataset_dir=None, output_size=224):
        if dataset_dir is None:
            dataset_dir = get_dataset_dir()
        self.image_dir = os.path.join(dataset_dir, PREPROC_DATASET_DIR, IMAGE_DIR)
        self.ann_dir = os.path.join(dataset_dir, PREPROC_DATASET_DIR, PREPROC_ANN_DIR)
        self.samples = get_samples(self.ann_dir)
        self.output_size = output_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        imagepath = os.path.join(self.image_dir, '{}.jpg'.format(filename))
        annpath = os.path.join(self.ann_dir, '{}.json'.format(filename))
        with open(imagepath, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        im = im.astype(np.float32)/255
        with open(annpath, 'r') as f:
            ann_data = json.load(f)
        is_single = ann_data['single']
        pose = np.array(ann_data['pose'], dtype=np.float32)
        bbx = ann_data['bbx']
        annotated = np.array(ann_data['annotated'],dtype=np.int)
        preproc, flip = get_preprocessing(bbx, is_single, self.output_size)
        im_warp = ImageTools.np_warp_im_bilinear(im, preproc, (self.output_size, self.output_size))
        pose_warp = preproc(pose.transpose()).transpose() - self.output_size/2
        if flip:
            flip_order = [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10]
            pose_flip = pose_warp[flip_order]
            annotated_flip = annotated[flip_order]
        else:
            pose_flip = pose_warp
            annotated_flip = annotated
        f = 300.0
        depth = 2400
        pos3d = np.zeros((28,3), dtype=np.float32)
        mpiinf_order = [25, 24, 23, 18,19,20,4,0,5, 7,16,15,14,9,10,11]
        pos3d[mpiinf_order,:2]=pose_flip*depth/f
        pos3d[:,2] = depth
        annotated = np.zeros((28),dtype=np.bool)
        annotated[mpiinf_order] = annotated_flip
        return im_warp.transpose(2,0,1), f, pos3d, annotated, False

def get_preprocessing(bbx, is_single, output_size):
    if is_single:
        max_translate = 0.1
        max_scale = np.log(1.25)-0.1
        min_scale = np.log(0.8)-0.1
    else:
        max_translate = 0.02
        max_scale = -0.08
        min_scale = -0.12
    max_rotate = 30*np.pi/180
    max_perspective_strength = 0.1
    flip = np.random.randint(2)

    scale = np.exp(np.random.uniform(min_scale, max_scale, size=(2)))
    scale_angle = np.random.uniform(0,np.pi*2)
    translate = np.random.uniform(-max_translate, max_translate, size=(2))
    rotation_angle = np.random.uniform(-max_rotate,max_rotate)
    mid_x = (bbx[0]+bbx[1])/2
    mid_y = (bbx[2]+bbx[3])/2
    perspective_angle = np.random.uniform(0, np.pi*2)
    perspective_strength = np.random.uniform(0, 0.1)
    r = np.sqrt((bbx[0]-mid_x)**2+(bbx[2]-mid_y)**2)
    transforms = [ImageTools.translation_as_affine([-mid_x, -mid_y])]
    transforms.append(ImageTools.scale_as_affine(0, (1/r, 1/r)))
    if flip:
        transforms.append(ImageTools.scale_as_affine(0, (-1, 1)))
    transforms.append(ImageTools.scale_as_affine(scale_angle, scale))
    transforms.append(ImageTools.rotate_as_affine(rotation_angle))
    transforms.append(ImageTools.translation_as_affine(translate))
    transforms.append(ImageTools.perspective_as_affine(perspective_angle, perspective_strength))
    transforms.append(ImageTools.scale_as_affine(0, (output_size/2, output_size/2)))
    transforms.append(ImageTools.translation_as_affine((output_size/2, output_size/2)))
    return ImageTools.stack_affine_transforms(transforms), flip


def get_samples(ann_dir):
    ann_files = os.listdir(ann_dir)
    ret = []
    for fname in ann_files:
        ret.append(int(fname[:-5]))
    return sorted(ret)


def main():
    x = Dataset()
    # x[4]
    for i in range(len(x)):
        x[i]

if __name__ == '__main__':
    main()


