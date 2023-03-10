import os
import json

from general_utils.environment_variables import get_dataset_dir
from Datasets.Coco.RawDataset import COCO_PREPROC_DIR_NAME
import ImageTools.ImageTools as ImageTools
import numpy as np
import cv2

class PreprocessedDataset():
    def __init__(self, samples, dataset_dir=None, output_size=224, augment=False):
        if dataset_dir is None:
            dataset_dir = get_dataset_dir()
        dataset_dir = os.path.join(dataset_dir, COCO_PREPROC_DIR_NAME)
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.annotations_dir = os.path.join(dataset_dir, 'anns')
        self.samples = samples
        self.output_size = output_size
        self.augment=augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        imagepath = os.path.join(self.image_dir, '{}.jpg'.format(filename))
        annpath = os.path.join(self.annotations_dir, '{}.json'.format(filename))
        with open(imagepath, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        im = im.astype(np.float32)/255
        with open(annpath, 'r') as f:
            ann_data = json.load(f)
        pose = np.array(ann_data['pose'], dtype=np.float32)
        valid = (np.array(ann_data['valid'], dtype=np.int) == 2).astype(np.int)
        bbx = ann_data['bbx']
        preproc, flip = get_preprocessing(bbx, self.output_size)
        im_warp = ImageTools.np_warp_im_bilinear(im, preproc, (self.output_size, self.output_size))
        pose_warp = preproc(pose.transpose()).transpose() - self.output_size/2
        if flip:
            flip_order = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
            pose_flip = pose_warp[flip_order]
            valid_flip = valid[flip_order]
        else:
            pose_flip = pose_warp
            valid_flip = valid
        # face keypoints not in mpii3dhp dataset
        mpiinf_order = [-1,-1,-1,-1,-1, 9, 14, 10,15,11,16,18,23,19,24,20,25]
        mpiinf_order = mpiinf_order[5:]
        f = 300.0
        depth = 2400
        pos3d = np.zeros((28,3), dtype=np.float32)
        # magical "5" to remove face keypoints without good match
        pos3d[mpiinf_order,:2]=pose_flip[5:]*depth/f
        pos3d[:,2] = depth
        valid = np.zeros((28),dtype=np.bool)
        valid[mpiinf_order] = valid_flip[5:]
        return im_warp.transpose(2,0,1), f, pos3d, valid, False


def get_preprocessing(bbx, output_size):
    # don't know if someone is close, better do small scale/translate
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




def get_normal():
    # sample 16971 failed to preprocess because it is a png file (with .jpg file ending)
    # I'm using jpegtran for perfect jpeg resize, I do not want to put in effort for this one sample
    train_idx = sorted(list(set(range(0, 97035))-{16971}))
    val_idx = list(range(97035, 101000))
    return PreprocessedDataset(train_idx+val_idx, augment=True)
