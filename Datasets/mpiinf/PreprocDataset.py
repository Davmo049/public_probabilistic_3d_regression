import argparse
import os
import json
import shutil

import numpy as np
import torch

from Datasets.mpiinf.RawDataset import MPIINF_PREPROCESSED_DIR
from Datasets.OpenImages.OpenImagesBackground import Dataset as BackgroundDataset
from general_utils.environment_variables import get_dataset_dir, get_cluster_dataset_dir
import Preprocessing
import ImageTools.ImageTools as ImageTools
import Losses # to compute bias

from PIL import Image

# cv2 fucks things up, include last
import cv2

# muz0 and D from "compute_muz0_D
muz0 = [3.980654943308963,
        3.907159922829376,
        4.033198150223853,
        4.033600982132145,
        4.2430324505749555,
        3.8311409703409463,
        3.752016154925238,
        3.6238805357100907,
        3.854972769175773,
        3.9434813557962323,
        4.31209934715221,
        4.356130080534959,
        4.344612859343316,
        3.8681606764766676,
        3.9026139804505435,
        4.330587336319432,
        4.219631417597879,
        4.194509199119434,
        4.29662906539081,
        4.601557646192438,
        4.693118144089817,
        4.579187381883249,
        4.562981632973335,
        4.299352866014174,
        4.65554883967894,
        4.626916162009388,
        4.588978556595649,
        4.586727013481686]

D = [2.7873109244153156,
     2.87822851693656,
     2.724107144330178,
     2.7236691165976428,
     2.6010759182828576,
     2.893657403752737,
     2.95177925399394,
     3.122440529947081,
     2.858683584743108,
     2.874764743309446,
     2.741624555396627,
     2.8152255004183733,
     2.874191557466427,
     2.8651945586648613,
     2.9201666544599374,
     2.869485467729028,
     3.1239231443535647,
     3.227254240536207,
     2.610964470979428,
     2.681270767879899,
     3.0253809254874433,
     3.0821430714766227,
     3.104561047202366,
     2.6589914891860493,
     2.676283086654554,
     2.9791529263792165,
     3.061008200562988,
     3.0730204513390413]


class PreprocessedDataset():
    def __init__(self, dataset_dir=None, output_size=224, use_noisy_bbx=False, augment=False, exclude_invalid_samples=True, load_from_frame_dump=False, load_from_crop_frame_dump=False, normalize_height=False):
        if dataset_dir is None:
            dataset_dir = get_dataset_dir()
        self.dir = os.path.join(dataset_dir, MPIINF_PREPROCESSED_DIR)
        print(self.dir)
        self.background_generator = BackgroundDataset(dataset_dir, output_size)
        self.load_from_frame_dump = load_from_frame_dump
        assert(not(load_from_frame_dump and load_from_crop_frame_dump))
        self.load_from_crop_frame_dump = load_from_crop_frame_dump
        self.samples = self.index_samples(exclude_invalid_samples)
        self.output_size = output_size
        self.use_noisy_bbx = use_noisy_bbx
        self.augment = augment
        self.normalize_height = normalize_height

    def index_samples(self, exclude_invalid_samples):
        subjects = sorted(list(map(int, os.listdir(self.dir))))
        if self.load_from_crop_frame_dump:
            subdir = 'frames_downsampled'
        else:
            subdir = 'frames'
        if self.load_from_crop_frame_dump or self.load_from_frame_dump:
            ret = []
            for subj in subjects:
                subj_dir = os.path.join(self.dir, str(subj))
                recordings = sorted(list(map(int, os.listdir(subj_dir))))
                for rec in  recordings:
                    rec_dir = os.path.join(subj_dir, str(rec))
                    camera_idxs = sorted(list(map(int, os.listdir(rec_dir))))
                    for cam_idx in camera_idxs:
                        cam_dir = os.path.join(rec_dir, str(cam_idx))

                        frames = os.listdir(os.path.join(cam_dir, subdir))
                        frames = sorted(list(map(lambda x: int(x[:-4]), frames)))
                        for frame in frames:
                            mask_path = os.path.join(cam_dir, 'mask', '{}.png'.format(frame))
                            if os.path.exists(mask_path):
                                ret.append((subj, rec, cam_idx, frame))
            return ret
        elif exclude_invalid_samples:
            ret = load_sample_list('valid_frames.csv')
        else:
            ret = []
            for subj in subjects:
                subj_dir = os.path.join(self.dir, str(subj))
                recordings = sorted(list(map(int, os.listdir(subj_dir))))
                for rec in  recordings:
                    rec_dir = os.path.join(subj_dir, str(rec))
                    camera_idxs = sorted(list(map(int, os.listdir(rec_dir))))
                    for cam_idx in camera_idxs:
                        cam_dir = os.path.join(rec_dir, str(cam_idx))
                        frames = os.listdir(os.path.join(cam_dir, 'frame_data'))
                        frames = sorted(list(map(lambda x: int(x[:-5]), frames)))
                        for frame in frames:
                            ret.append((subj, rec, cam_idx, frame))
        return ret

    def __getitem__(self, idx):
        subject, recording, cam, frame = self.samples[idx]
        image = self.load_image(subject, recording, cam, frame)
        image = image.astype(np.float32)/255
        frame_data, calib_data = self.load_data(subject, recording, cam, frame)
        bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable = frame_data['augmentable']
        if self.use_noisy_bbx:
            bounding_box = np.array(frame_data['bbx_noise'])
        else:
            bounding_box = np.array(frame_data['bbx_ideal'])
        if self.load_from_frame_dump:
            bbx_crop = frame_data['bbx_crop']
            bounding_box[0] -= bbx_crop[0]
            bounding_box[1] -= bbx_crop[0]
            bounding_box[2] -= bbx_crop[2]
            bounding_box[3] -= bbx_crop[2]
            I = calib_data['I']
            I[0][2] = I[0][2] - bbx_crop[0]
            I[1][2] = I[1][2] - bbx_crop[2]
        elif self.load_from_crop_frame_dump:
            bbx_ds = frame_data['bbx_ds']
            ds_r = frame_data['downsample_rate']
            bounding_box[0] = (bounding_box[0] - (bbx_ds[0] - (ds_r-1)/2)) / ds_r
            bounding_box[1] = (bounding_box[1] - (bbx_ds[0] - (ds_r-1)/2)) / ds_r
            bounding_box[2] = (bounding_box[2] - (bbx_ds[2] - (ds_r-1)/2)) / ds_r
            bounding_box[3] = (bounding_box[3] - (bbx_ds[2] - (ds_r-1)/2)) / ds_r
            I = calib_data['I']
            I[0][2] = I[0][2] - bbx_ds[0] - (ds_r-1)/2
            I[1][2] = I[1][2] - bbx_ds[2] - (ds_r-1)/2
            I[0][0] = I[0][0] / ds_r
            I[0][1] = I[0][1] / ds_r
            I[0][2] = I[0][2] / ds_r
            I[1][0] = I[1][0] / ds_r
            I[1][1] = I[1][1] / ds_r
            I[1][2] = I[1][2] / ds_r

        to_net_input = self.get_transform(bounding_box, calib_data, self.output_size, self.augment)
        image_mapping, real_cam_to_ideal_cam, ideal_intrinsic, flip = to_net_input
        output_size = [self.output_size, self.output_size]
        if bg_augmentable:
            mask = self.load_mask(subject, recording, cam, frame)
            if self.load_from_frame_dump:
                mask_adj = mask[bbx_crop[2]:bbx_crop[3], bbx_crop[0]:bbx_crop[1]]
                alpha = np.logical_not(mask_adj == 1)
            elif self.load_from_crop_frame_dump:
                alpha = mask.astype(np.float32) / 128
            else:
                alpha = np.logical_not(mask == 1)
            image_alpha = np.empty((image.shape[0], image.shape[1], 4), dtype=np.float32)
            image_alpha[:,:,:3] = image*alpha.reshape(image.shape[0], image.shape[1], 1)
            image_alpha[:,:,3] = alpha
            image_alpha_input_space = ImageTools.np_warp_im_bilinear(image_alpha, image_mapping, output_size)
            del image, alpha, image_alpha
            image_input_space = image_alpha_input_space[:,:,:3]
            alpha_input_space = image_alpha_input_space[:,:,3]
            del image_alpha_input_space
            if self.augment:
                background_im = self.background_generator.random_sample()
            else:
                background_im = self.background_generator.getitem_no_aug(idx%len(self.background_generator))
            image_input_space += background_im * (1-alpha_input_space.reshape(output_size[0], output_size[1], 1))
            del background_im
            del alpha_input_space
        else:
            image_input_space = ImageTools.np_warp_im_bilinear(image, image_mapping, output_size)
        pos3d = np.array(frame_data['pos3d'])
        if self.normalize_height:
            height = compute_height(pos3d)
            pos3d_old = np.copy(pos3d)
            pos3d *= 1715/height
        if flip:
            pos3d_flipped = np.empty((28,3), dtype=np.float32)
            pos3d_flipped[:8] = pos3d[:8] # spine/neck/head
            pos3d_flipped[8:13] = pos3d[13:18] # arm/hand
            pos3d_flipped[13:18] = pos3d[8:13] # arm/hand
            pos3d_flipped[18:23] = pos3d[23:] # leg/foot
            pos3d_flipped[23:] = pos3d[18:23] # leg/foot
        else:
            pos3d_flipped = pos3d
        R = real_cam_to_ideal_cam[:3,:3]
        pos3d_in_ideal_cam = np.array(list(map(lambda p: np.matmul(R, p), pos3d_flipped))).astype(np.float32)
        image_input_space = image_input_space.astype(np.float32).transpose(2,0,1)
        annotated = np.ones((28), dtype=np.bool)
        valid_depth = True
        return image_input_space, ideal_intrinsic[0,0], pos3d_in_ideal_cam, annotated, valid_depth
        # pos2dH = pos2d_in_ideal_cam = list(map(lambda p: np.matmul(ideal_intrinsic, p), pos3d_in_ideal_cam))
        # pos2dH = np.array(pos2dH)
        # print(pos2dH.shape)
        # pos2d = pos2dH[:, :2]/pos2dH[:,2].reshape(-1, 1)
        # plt.plot(pos2d[:,0], pos2d[:,1], 'rx')

    @staticmethod
    def get_transform(bounding_box, calib_data, output_size, augment):
        translate_strength = 0.1
        rotation_strength = 0.1
        max_upscale = np.log(1.25)-0.1
        max_downscale = np.log(0.8)-0.1
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
        if augment:
            # depth not well defined for differnet scaling for different axis or projective
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
        else:
            flip = False
            scale = np.exp((max_downscale+max_upscale)/2)
            scale = np.array([scale, scale])
            scale_center = np.array([1,1.0])*output_size/2
            transforms.append(ImageTools.scale_as_affine(0, scale, scale_center=scale_center))
            ideal_intrinsic = np.copy(ideal_intrinsic)
            ideal_intrinsic[0,0] *= scale[0]
            ideal_intrinsic[1,1] *= scale[0]
        if flip:
            real_cam_to_ideal_cam[0] *= -1
        image_mapping = ImageTools.stack_affine_transforms(transforms)
        return image_mapping, real_cam_to_ideal_cam, ideal_intrinsic, flip


    def load_mask(self, subject, recording, cam, frame):
        if self.load_from_crop_frame_dump:
            path = os.path.join(self.dir, str(subject), str(recording), str(cam), 'alpha_downsampled', '{}.png'.format(frame))
        else:
            path = os.path.join(self.dir, str(subject), str(recording), str(cam), 'mask', '{}.png'.format(frame))
        with open(path, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return im

    def load_image(self, subject, recording, cam, frame):
        if self.load_from_frame_dump:
            frame_path = os.path.join(self.dir, str(subject), str(recording), str(cam), 'frames', '{}.jpg'.format(frame))
            with open(frame_path, 'rb') as f:
                data = f.read()
            buf = np.frombuffer(data, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return im[:,:,::-1]
        elif self.load_from_crop_frame_dump:
            frame_path = os.path.join(self.dir, str(subject), str(recording), str(cam), 'frames_downsampled', '{}.jpg'.format(frame))
            with open(frame_path, 'rb') as f:
                data = f.read()
            buf = np.frombuffer(data, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return im[:,:,::-1]

        else:
            vidpath = os.path.join(self.dir, str(subject), str(recording), str(cam), 'vid.avi')
            stream = cv2.VideoCapture(vidpath)
            stream.set(1, frame)
            success, image = stream.read()
            stream.release()
            if not success:
                raise Exception('could not parse image from video: {} {} {} [}'.format(subject, recording, cam, frame))
            image = image[:,:,::-1]
            return image

    def load_data(self, subject, recording, cam, frame):
        frame_data_path = os.path.join(self.dir, str(subject), str(recording), str(cam), 'frame_data', '{}.json'.format(frame))
        calib_path = os.path.join(self.dir, str(subject), str(recording), str(cam), 'calib.json')
        with open(frame_data_path, 'r') as f:
            frame_data = json.load(f)
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        return frame_data, calib_data

    def __len__(self):
        return len(self.samples)


def compute_invalid_frames(preproc_dataset):
    valid_frames = []
    invalid_frames = []
    for idx, sample_idx in enumerate(preproc_dataset.samples):
        subject_idx, recording_idx, camera_idx, frame = sample_idx
        frame_data, _ = preproc_dataset.load_data(subject_idx, recording_idx, camera_idx, frame)
        width = 2048
        height = 2048
        bbx = frame_data['bbx_ideal']
        valid = (bbx[0] >= 0) & (bbx[1] <= width) & (bbx[2] >= 0) & (bbx[3] <= height)
        if valid:
            valid_frames.append(sample_idx)
        else:
            invalid_frames.append(sample_idx)
    return valid_frames, invalid_frames


def dump_sample_list(samples, filename):
    with open(filename, 'w') as f:
        for i in range(len(samples)-1):
            s = samples[i]
            f.write('{} {} {} {}\n'.format(s[0], s[1], s[2], s[3]))
        f.write('{} {} {} {}'.format(s[0], s[1], s[2], s[3]))


def load_sample_list(filename):
    samples = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            while l[-1] == '\n':
                l = l[:-1]
            samples.append(tuple(map(int, l.split(' '))))
    return samples


def compute_valid_frames():
    dataset = PreprocessedDataset(exclude_invalid_samples=False)
    valid_frames, invalid_frames = compute_invalid_frames(dataset)
    dump_sample_list(valid_frames, 'valid_frames.csv')
    dump_sample_list(invalid_frames, 'invalid_frames.csv')


def compute_muz0_D():
    preproc_dataset = PreprocessedDataset(augment=False, exclude_invalid_samples=True)
    output_size = 224
    S = output_size
    zs = []
    for idx, sample_idx in enumerate(preproc_dataset.samples):
        subject_idx, recording_idx, camera_idx, frame = sample_idx
        frame_data, calib_data = preproc_dataset.load_data(subject_idx, recording_idx, camera_idx, frame)
        bounding_box = np.array(frame_data['bbx_ideal'])
        intrinsic = np.array(calib_data['I'])
        r = Preprocessing.get_transform_to_ideal_camera(output_size, bounding_box, intrinsic, desired_up=None) # up in cam = None -> no change
        real_cam_to_ideal_cam, ideal_intrinsic, cam_im_to_ideal_im = r
        f = ideal_intrinsic[0,0] # focal length in pixels
        pos3d = np.array(frame_data['pos3d'])
        R = real_cam_to_ideal_cam[:3,:3]
        pos3d_in_ideal_cam = list(map(lambda i: np.matmul(R, pos3d[i,:]), range(pos3d.shape[0])))
        z_in_ideal_cam = np.array(list(map(lambda x: x[2], pos3d_in_ideal_cam)))
        zp = z_in_ideal_cam/f
        zs.append(zp)
    zs = np.array(zs)
    for i in range(28):
        mu0 = np.sqrt(np.max(zs[:,i])*np.min(zs[:,i]))
        print('idx: {}'.format(i))
        print(mu0)
        D = np.sqrt(np.max(zs[:,i])/np.min(zs[:,i]))
        print(D)


def compute_zp():
    preproc_dataset = PreprocessedDataset(augment=False, exclude_invalid_samples=True)
    output_size = 224
    S = output_size
    zps = []
    for idx, sample_idx in enumerate(preproc_dataset.samples):
        subject_idx, recording_idx, camera_idx, frame = sample_idx
        frame_data, calib_data = preproc_dataset.load_data(subject_idx, recording_idx, camera_idx, frame)
        bounding_box = np.array(frame_data['bbx_ideal'])
        intrinsic = np.array(calib_data['I'])
        r = Preprocessing.get_transform_to_ideal_camera(output_size, bounding_box, intrinsic, desired_up=None) # up in cam = None -> no change
        real_cam_to_ideal_cam, ideal_intrinsic, cam_im_to_ideal_im = r
        f = ideal_intrinsic[0,0] # focal length in pixels
        pos3d = np.array(frame_data['pos3d'])
        R = real_cam_to_ideal_cam[:3,:3]
        pos3d_in_ideal_cam = list(map(lambda i: np.matmul(R, pos3d[i,:]), range(pos3d.shape[0])))
        z_in_ideal_cam = np.array(list(map(lambda x: x[2], pos3d_in_ideal_cam)))
        zp = np.log(z_in_ideal_cam*S/f)
        zps.append(zp)
    zps = np.array(zps)
    return zps.tranpose()

def compute_height(gt):
    path = [17,16,15,14,13,5,8,9,10,11,12]

    start = path[:-1]
    end = path[1:]
    height = np.sum(np.linalg.norm(gt[end]-gt[start], axis=1))
    return height

def compute_prior_output():
    should_compute_zp = False
    if should_compute_zp:
        zps = compute_zp()
        for i in range(28):
            os.makedirs('zvals_mpiinf')
            with open('zvals_mpiinf/{}_zp.txt'.format(i), 'w') as f:
                zss = list(zps[i].reshape(-1))
                zstr = ','.join(map(str, zss))
                f.write(zstr)
    else:
        zps = []
        for i in range(28):
            with open('zvals_mpiinf/{}_zp.txt'.format(i), 'r') as f:
                zp = list(map(float, f.read().split(',')))
            zps.append(zp)
    for i in range(28):
        zpmin = np.min(zps[i])
        zpmax = np.max(zps[i])
        log_mu0 = (zpmax+zpmin)/2
        D = (zpmax-zpmin)/2
        znorm = (zps-log_mu0)/D
        bnu = optimize_bnu(znorm[::100])
        bnu = optimize_bnu(znorm, bnu)
        distance_remapper = Losses.DistanceRemapper(torch.tensor((D)))
        bias = distance_remapper.invert(bnu, 0)
        print('i: {}'.format(i))
        print('bnu: {}'.format(bnu))
        print('bias: {}'.format(bias))

def optimize_bnu(z, bnu=None):
    if bnu is None:
        bnu = np.array((1.0, 0.0))
    steplen = 1.0
    while steplen != 0:
        bnu, steplen, fv = line_search(bnu, z, steplen)
    return bnu

def line_search(bnu, z, steplen):
    fbnu = f(bnu[0], z, bnu[1])
    dfbnu = df(bnu[0], z, bnu[1])
    dfbnun = dfbnu/np.linalg.norm(dfbnu)
    steplen *= 3
    while steplen > 1e-4:
        testv = bnu - dfbnun*steplen
        fbnu_test = f(testv[0], z, testv[1])
        diff = fbnu_test-fbnu
        if diff < 0:
            retbnu = bnu-dfbnun*steplen/2
            fbnu_test = f(retbnu[0], z, retbnu[1])
            return retbnu, steplen/2, fbnu_test
        steplen /= 2
    return bnu, 0, fbnu

def construct_znorm_from_zp(zp, D, mu0):
    znorm = (zp-np.log(mu0))/D
    return znorm

def h(b,z,nu):
    absv = np.abs(b*z-nu)
    mask = absv > 1
    ret = absv**2/2
    ret[mask] = absv[mask]-1/2
    return ret

def f(b,z,nu):
    return -np.log(b)+np.mean(np.exp(h(b,z,nu)))

def df(b,z,nu):
    dfdb = -1/b+np.mean(z*dh(b,z,nu)*np.exp(h(b,z,nu)))
    dfdnu = -np.mean(dh(b,z,nu)*np.exp(h(b,z,nu)))
    return np.array((dfdb, dfdnu))

def dh(b,z,nu):
    v = b*z-nu
    mask = np.abs(v) > 1
    ret = v
    ret[mask] = np.sign(v[mask])
    return ret

def create_downsampled_dump(subj):
    dataset_dir = get_dataset_dir()
    mpiinfdir = os.path.join(dataset_dir, MPIINF_PREPROCESSED_DIR)
    subj_dir = os.path.join(mpiinfdir, str(subj))
    recs = sorted(os.listdir(subj_dir))
    valid_frames = set(load_sample_list('valid_frames.csv'))
    for rec in recs:
        rec_dir = os.path.join(subj_dir, rec)
        cams = sorted(os.listdir(rec_dir))
        for cam in cams:
            cam_dir = os.path.join(rec_dir, cam)
            video_file = os.path.join(cam_dir, 'vid.avi')
            mask_dir = os.path.join(cam_dir, 'mask')
            frame_dump = os.path.join(cam_dir, 'frames_downsampled')
            alpha_dump = os.path.join(cam_dir, 'alpha_downsampled')
            if os.path.exists(frame_dump):
                shutil.rmtree(frame_dump)
            if os.path.exists(alpha_dump):
                shutil.rmtree(alpha_dump)
            os.makedirs(frame_dump)
            os.makedirs(alpha_dump)
            dump_every_n_th_frame = 2
            vidpath = os.path.join(video_file)
            stream = cv2.VideoCapture(vidpath)
            frame = 0
            time_since_last_frame = np.inf
            while True:
                cur_sample = (int(subj), int(rec), int(cam), int(frame))

                success, image = stream.read()
                if not(success):
                    break
                image = image[:,:,::-1]
                if (cur_sample in valid_frames) and (time_since_last_frame+1 >= dump_every_n_th_frame):
                    mask_file = os.path.join(mask_dir, '{}.png'.format(frame))
                    if not(os.path.exists(mask_file)):
                        continue
                    with open(mask_file, 'rb') as f:
                        mask_data = f.read()
                    mask_buf = np.frombuffer(mask_data, dtype=np.uint8)
                    mask = cv2.imdecode(mask_buf, cv2.IMREAD_GRAYSCALE)
                    jsonfile = os.path.join(cam_dir, 'frame_data', '{}.json'.format(frame))
                    with open(jsonfile, 'r') as f:
                        frame_data = json.load(f)
                    bbx = frame_data['bbx_ideal']
                    bbx_ds, ds_r = get_downsample_bbx(bbx, image.shape)
                    assert(ds_r < 16)
                    if ds_r == 1:
                        downsampled_image = image[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1],:]
                        mask_adj = mask[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1]]
                        alpha = np.logical_not(mask_adj == 1).astype(np.uint8)*128
                        im_save = downsampled_image
                    else:
                        image = image[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1],:].astype(np.uint16)
                        mask_adj = mask[bbx_ds[2]:bbx_ds[3], bbx_ds[0]:bbx_ds[1]]
                        alpha = np.logical_not(mask_adj == 1).astype(np.uint16)
                        image_alpha = image*alpha.reshape(image.shape[0], image.shape[1], 1)
                        downsampled_image = image[::ds_r, ::ds_r]
                        downsampled_image_alpha = image_alpha[::ds_r, ::ds_r]
                        downsampled_alpha = alpha[::ds_r, ::ds_r]
                        for i in range(ds_r):
                            for j in range(ds_r):
                                if i == 0 and j == 0:
                                    continue
                                downsampled_image += image[i::ds_r, j::ds_r]
                                downsampled_image_alpha += image_alpha[i::ds_r, j::ds_r]
                                downsampled_alpha += alpha[i::ds_r, j::ds_r]
                        alpha = (downsampled_alpha * 128 + ((ds_r**2)//2))//(ds_r**2)
                        dsa = downsampled_alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
                        with np.errstate(divide='ignore'):
                            image_alpha = (downsampled_image_alpha+dsa//2)//dsa
                        image = (downsampled_image+((ds_r**2)//2))//(ds_r**2)
                        image_alpha[alpha == 0] = image[alpha == 0]
                        im_save = image_alpha.astype(np.uint8)
                        alpha = alpha.astype(np.uint8)
                    im_save = Image.fromarray(im_save)
                    save_name = os.path.join(frame_dump, '{}.jpg'.format(str(frame)))
                    im_save.save(save_name)
                    im_save = Image.fromarray(alpha)
                    save_name = os.path.join(alpha_dump, '{}.png'.format(str(frame)))
                    im_save.save(save_name)

                    frame_data['bbx_ds'] = list(map(int, bbx_ds))
                    frame_data['downsample_rate'] = int(ds_r)
                    with open(jsonfile,'w') as f:
                        json.dump(frame_data, f)
                    time_since_last_frame = 0
                else:
                    time_since_last_frame += 1
                frame+=1
            stream.release()

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



def split_video_into_images(subj):
    dataset_dir = get_dataset_dir()
    mpiinfdir = os.path.join(dataset_dir, MPIINF_PREPROCESSED_DIR)
    subj_dir = os.path.join(mpiinfdir, str(subj))
    recs = sorted(os.listdir(subj_dir))
    valid_frames = set(load_sample_list('valid_frames.csv'))
    for rec in recs:
        rec_dir = os.path.join(subj_dir, rec)
        cams = sorted(os.listdir(rec_dir))
        for cam in cams:
            cam_dir = os.path.join(rec_dir, cam)
            video_file = os.path.join(cam_dir, 'vid.avi')
            frame_dump = os.path.join(cam_dir, 'frames')
            os.makedirs(frame_dump)
            dump_every_n_th_frame = 2
            vidpath = os.path.join(video_file)
            stream = cv2.VideoCapture(vidpath)
            frame = 0
            time_since_last_frame = np.inf
            while True:
                cur_sample = (int(subj), int(rec), int(cam), int(frame))

                success, image = stream.read()
                if not(success):
                    break
                if (cur_sample in valid_frames) and (time_since_last_frame+1 >= dump_every_n_th_frame):
                    jsonfile = os.path.join(cam_dir, 'frame_data', '{}.json'.format(frame))
                    with open(jsonfile, 'r') as f:
                        frame_data = json.load(f)
                    bbx = frame_data['bbx_ideal']
                    start_x = int(bbx[0])
                    end_x = int(bbx[1])
                    start_y = int(bbx[2])
                    end_y = int(bbx[3])
                    w = (end_x-start_x)*1.3
                    h = (end_y-start_y)*1.3
                    h = max(w,h)
                    w = h
                    mid_x, mid_y = (start_x+end_x)/2, (start_y+end_y)/2
                    start_crop_x = max(0, int(mid_x-w/2))
                    end_crop_x = min(image.shape[1], int(mid_x+w/2))
                    start_crop_y = max(0, int(mid_y-h/2))
                    end_crop_y = min(image.shape[0], int(mid_y+h/2))
                    im_save = image[start_crop_y:end_crop_y, start_crop_x:end_crop_x, ::-1]
                    im_save = Image.fromarray(im_save)
                    save_name = os.path.join(frame_dump, '{}.jpg'.format(str(frame)))
                    im_save.save(save_name)

                    bbx_crop = [start_crop_x, end_crop_x, start_crop_y, end_crop_y]
                    frame_data['bbx_crop'] = bbx_crop
                    with open(jsonfile,'w') as f:
                        json.dump(frame_data, f)
                    time_since_last_frame = 0
                else:
                    time_since_last_frame += 1
                frame+=1
            stream.release()

def main_split_video_into_images():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--target_subjects', type=int, default=0)
    args = arg_parser.parse_args()
    target_subjects = args.target_subjects
    split_video_into_images(target_subjects)

def visualize_height():
    preproc_dataset = PreprocessedDataset(augment=False, exclude_invalid_samples=True)
    h = []
    for idx, sample_idx in enumerate(preproc_dataset.samples[::100]):
        subject_idx, recording_idx, camera_idx, frame = sample_idx
        frame_data, calib_data = preproc_dataset.load_data(subject_idx, recording_idx, camera_idx, frame)
        pos3d = np.array(frame_data['pos3d'])
        height = compute_height(pos3d)
        h.append(height)
    print(np.mean(h))


def main():
    pass
    # compute_valid_frames()
    compute_muz0_D()
    # compute_prior_output()
    # main_split_video_into_images()
    visualize_height()


    # dataset = PreprocessedDataset(exclude_invalid_samples=True, load_from_frame_dump=False)
    # print(len(dataset))
    # for i in range(0, len(dataset), 1):
    #     image_input_space, ideal_intrinsic, pos3d_in_ideal_cam = dataset[i]

if __name__ == '__main__':
    main()
