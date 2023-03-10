import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image

from general_utils import environment_variables
import ImageTools.ImageTools as ImageTools

#cv2 fucks things up, include last
import cv2

class AugParam():
    def __init__(self, enable_flip, max_rot, max_scale):
        self.enable_flip = enable_flip
        self.max_rot = max_rot*np.pi/180
        self.max_scale = max_scale

class Dataset():
    def __init__(self, dataset_dir=None, output_size=224):
        if dataset_dir is None:
            dataset_dir = environment_variables.get_dataset_dir()
        dataset_dir = os.path.join(dataset_dir, 'OpenImageDump', 'bg_preprocess')
        samples = sorted(os.listdir(dataset_dir))
        self.dataset_dir = dataset_dir
        self.samples = samples
        self.len = len(self.samples)
        self.aug = AugParam(True, 30, 2.0)
        self.out_size = output_size

    def __getitem__(self, idx):
        return self.getitem_with_aug(idx, self.aug)

    def getitem_with_aug(self, idx, aug):
        impath = os.path.join(self.dataset_dir, self.samples[idx])
        with open(impath, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        im = im.astype(np.float32) / 255
        if aug.enable_flip:
            flip = np.random.randint(2) == 1
        else:
            flip = 0
        w,h = im.shape[1], im.shape[0]
        rotation_angle = (np.random.uniform()*2-1)*aug.max_rot
        base_scale = 1
        scale = base_scale * np.random.uniform(1, aug.max_scale)
        scale = max(scale, self.out_size/w, self.out_size/h)
        crop_size_orig_im = self.out_size/scale
        w_c = w-crop_size_orig_im
        h_c = h-crop_size_orig_im
        x_center = np.random.uniform()*w_c
        y_center = np.random.uniform()*h_c
        new_center = np.array([x_center, y_center])
        transforms = []
        if flip:
            transforms.append(ImageTools.fliplr_as_affine(im.shape))
        transforms.append(ImageTools.translation_as_affine(-new_center))
        # move center to origin
        # scale box to -1, 1
        transforms.append(ImageTools.rotate_as_affine(rotation_angle))
        transforms.append(ImageTools.scale_as_affine(0, np.array([scale, scale])))

        # finish
        transform = ImageTools.stack_affine_transforms(transforms)
        net_in_image = ImageTools.np_warp_im_bilinear(im, transform, (self.out_size, self.out_size))
        return net_in_image

    def __len__(self):
        return self.len

    def getitem_no_aug(self, idx):
        no_aug = AugParam(False, 0, 1)
        return self.getitem_with_aug(idx, no_aug)
        
    def random_sample(self):
        i = np.random.randint(0, len(self))
        return self[i]

def downscale_dataset(dataset_dir=None, minsize=224):
    minsize = 224
    if dataset_dir is None:
        dataset_dir = environment_variables.get_dataset_dir()
    dataset_in_dir = os.path.join(dataset_dir, 'OpenImageDump', 'bg')
    dataset_out_dir = os.path.join(dataset_dir, 'OpenImageDump', 'bg_preprocess')
    if not(os.path.exists(dataset_out_dir)):
        os.makedirs(dataset_out_dir)
    samples = sorted(os.listdir(dataset_in_dir))
    for idx in tqdm.tqdm(range(len(samples))):
        impath = os.path.join(dataset_in_dir, samples[idx])
        outpath = os.path.join(dataset_out_dir, '{}.jpg'.format(idx))
        with open(impath, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            im = im[:,:,::-1]

            w,h = im.shape[1], im.shape[0]
            scale = max(1/8, minsize/w, minsize/h)
            if scale < 1:
                im = im.astype(np.int32)
                if scale < 0.5:
                    num_downsamples = int(np.log2(0.5/scale))
                    pixel_ds = 2**num_downsamples
                    pad_w = (-w) % pixel_ds
                    pad_left = 0
                    pad_right = pad_w
                    pad_height = (-h) % pixel_ds
                    pad_top = 0
                    pad_bot = pad_height
                    im = np.pad(im, ((pad_top, pad_bot), (pad_left, pad_right), (0,0)), 'edge')
                    for _ in range(num_downsamples):
                        im = im[::2,::2] + im[::2, 1::2] + im[1::2, ::2] + im[1::2, 1::2]
                    div = 4**num_downsamples
                im = ((im + (div//2)) / div).astype(np.float32)
                w_cur, h_cur = im.shape[1], im.shape[0]
                # minor aspect ratio change from rounding
                scale_new = scale*pixel_ds
                c_pixels, r_pixels = int(scale_new*w_cur+0.5), int(scale_new*h_cur+0.5)
                cNew, rNew = np.meshgrid(np.arange(c_pixels)/scale_new, np.arange(r_pixels)/scale_new)
                im = ImageTools.np_interpolate_image(im, rNew, cNew)
                # 0.5 to round not floor
                im = (np.clip(np.round(im+0.5),0,255)).astype(np.uint8)
                im = Image.fromarray(im)
                im.save(outpath)
            else:
                _, extention = os.path.splitext(impath)
                outpath = os.path.join(dataset_out_dir, '{}'.format(idx)+extention)
                shutil.copyfile(impath, outpath)

if __name__ == '__main__':
    downscale_dataset(dataset_dir=None, minsize=224)
    x = Dataset()
    for i in range(100):
        x[i]
