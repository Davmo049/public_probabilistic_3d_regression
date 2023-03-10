import numpy as np
import torch
import os
import json

from general_utils.environment_variables import get_dataset_dir

import cv2

SYNTHETIC_DATA_PATH = 'Synthetic'
SUBSET_DIR = 'default'

class Dataset():
    def __init__(self, datasetdir=None, normalize_image=True):
        if datasetdir is None:
            datasetdir = os.path.join(get_dataset_dir(), SYNTHETIC_DATA_PATH, SUBSET_DIR)
        self.datasetdir = datasetdir
        self.num_samples = self.get_num_samples(datasetdir)
        self.normalize_image = normalize_image

    def get_train(self):
        train_samples = list(range(int(self.num_samples*0.8)))
        return DatasetInstance(self.datasetdir, train_samples, self.normalize_image)

    def get_eval(self):
        eval_samples = list(range(int(self.num_samples*0.8), self.num_samples))
        return DatasetInstance(self.datasetdir, eval_samples, self.normalize_image)

    @staticmethod
    def get_num_samples(directory):
        im_name_set = set(os.listdir(os.path.join(directory, 'images')))
        sample_name_set = set(os.listdir(os.path.join(directory, 'scenes')))
        assert(len(im_name_set) == len(sample_name_set))
        for i in range(len(im_name_set)):
            expected_imname = 'sample{:06d}.png'.format(i)
            if not(expected_imname in im_name_set):
                print(expected_imname)
                exit(0)
            expected_samplename = 'sample{:06d}.json'.format(i)
            if not(expected_samplename in sample_name_set):
                print(sample_name_set)
                exit(0)
        return len(im_name_set)


class DatasetInstance():
    def __init__(self, datasetdir, samples,normalize_image=True):
        self.mu0, self.D = self.find_mu_D()
        self.datasetdir = datasetdir
        self.samples = samples
        if normalize_image:
            self.mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
            self.std = np.array([0.229, 0.224, 0.225]).reshape(1,1, 3)
        else:
            self.mean = np.array([0.0,0,0]).reshape(1,1,3)
            self.std = np.array([1.0,1,1]).reshape(1,1,3)

    def find_mu_D(self):
        D = np.sqrt(5/3*1200/2000)
        mu0 = np.sqrt(3*5/1200/2000)
        return mu0, D

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = self.samples[idx]
        imname = os.path.join(self.datasetdir, 'images', 'sample{:06d}.png'.format(idx))
        with open(imname, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        jsonname = os.path.join(self.datasetdir, 'scenes', 'sample{:06d}.json'.format(idx))
        with open(jsonname, 'r') as f:
            scene = json.load(f)
        v = np.array(scene['pos_main'])
        f = scene['focal_length']
        v[1] = -v[1]
        im = im/255.0
        im = (im-self.mean)/self.std
        return im.transpose(2,0,1).astype(np.float32), f, v.reshape(1,3).astype(np.float32), np.ones((1),dtype=np.int), True


if __name__ == '__main__':
    main()
