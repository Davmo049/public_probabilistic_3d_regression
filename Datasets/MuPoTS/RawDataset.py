import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import os
import json

from general_utils.environment_variables import get_dataset_dir

import cv2

MUPO3D_PATH = 'MultiPersonTestSet'

def get_intrinsic_camera(rec_idx):
    recording = os.path.join(get_dataset_dir(), MUPO3D_PATH, 'TS{}'.format(rec_idx))
    p2 = []
    p3 = []
    matdump = os.path.join(recording, 'annot.mat')
    annot = scipy.io.loadmat(matdump)
    annot = annot['annotations']
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            a = annot[i,j]
            p2_c = list(a['annot2'][0,0].transpose())
            p3_c = list(a['annot3'][0,0].transpose())
            # some keypoints project incorrectly don't know why
            p2 += p2_c[:8]
            p2 += p2_c[9:11]
            p2 += p2_c[12:15]
            p2 += p2_c[12:15]
            p2 += p2_c[16:]

            p3 += p3_c[:8]
            p3 += p3_c[9:11]
            p3 += p3_c[12:15]
            p3 += p3_c[12:15]
            p3 += p3_c[16:]
            # end of remove bad keypoints
    I = cam_calib(p2, p3)
    assert(np.abs(I[1,0]) < 1e-3)
    assert(np.abs(I[0,1]) < 1e-3)
    assert(np.abs(I[2,0]) < 1e-3)
    assert(np.abs(I[2,1]) < 1e-3)
    I[1,0] = 0
    I[0,1] = 0
    I[2, :2] = 0
    return I

def extract_bounding_boxes():
    # format of ret
    # [recording, frame, bbx_x_min, bbx_x_max, bbx_y_min, bbx_y_max]
    ret = []
    valid_count = 0
    invalid_count = 0
    for recording_idx in range(1,21):
        recording = os.path.join(get_dataset_dir(), MUPO3D_PATH, 'TS{}'.format(recording_idx))
        matdump = os.path.join(recording, 'annot.mat')
        annotations = scipy.io.loadmat(matdump)
        annotations = annotations['annotations']
        for frame in range(annotations.shape[0]):
            for person_idx in range(annotations.shape[1]):
                annotations_frame = annotations[frame,person_idx]
                valid = annotations_frame['isValidFrame'][0,0][0,0]
                if valid:
                    valid_count += 1
                    points2d = annotations_frame['annot2'][0,0]
                    min_x = np.min(points2d[0])
                    max_x = np.max(points2d[0])
                    min_y = np.min(points2d[1])
                    max_y = np.max(points2d[1])
                    points3d = annotations_frame['annot3'][0,0]
                    ret.append({'input': [recording_idx, frame, min_x, max_x, min_y, max_y], 
                                'gt': list(map(list, points3d))})
                else:
                    invalid_count += 1
    print('valid samples: {}'.format(valid_count))
    print('invalid samples: {}'.format(invalid_count))
    return ret

def cam_calib(p2, p3):
    # both Nx2/Nx3
    p2 = np.array(p2).copy()
    p3 = np.array(p3)
    P = np.zeros((p3.shape[0]*2, 8))
    t = np.empty((p3.shape[0]*2))
    norm2 = np.mean(np.abs(p2), axis=0)
    norm3 = np.mean(np.abs(p3), axis=0)
    for n in range(p2.shape[0]):
        P[2*n, :3] = p3[n]/norm3
        P[2*n, 6:] = -p3[n,:2]*p2[n,0]/norm3[:2]/norm2[0]
        t[2*n] = p3[n,2]*p2[n,0]
        P[2*n+1, 3:6] = p3[n]/norm3
        P[2*n+1, 6:] = -p3[n,:2]*p2[n,1]/norm3[:2]/norm2[1]
        t[2*n+1] = p3[n,2]*p2[n,1]
    sol = np.linalg.lstsq(P, t, rcond=None)[0]
    sol[:3] /= norm3
    sol[3:6] /= norm3
    sol[6:8] /= norm3[:2]
    sol[6:8] /= norm2
    I = np.ones((9))
    I[:-1] = sol
    I = I.reshape(3,3)
    return I


if __name__ == '__main__':
    annotations = extract_bounding_boxes()
    ann_dump = os.path.join(get_dataset_dir(), MUPO3D_PATH, 'ann.json')
    with open(ann_dump, 'w') as f:
        json.dump(annotations, f)
    intrinsics = []
    for i in range(1,21):
        cur_i = get_intrinsic_camera(i)
        cur_i_list = list(map(list, cur_i))
        intrinsics.append(cur_i_list)
    intrinsics_dump = os.path.join(get_dataset_dir(), MUPO3D_PATH, 'intrinsics.json')
    with open(intrinsics_dump, 'w') as f:
        json.dump(intrinsics, f)
