import argparse
import json
import os
import shutil

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image

from general_utils.environment_variables import get_dataset_dir, get_cluster_dataset_dir

# cv2 fucks things up, include last
import cv2

MPIINFDIR = 'MPIINF'

def RawDatasetEval():
    pass

def get_matlab_stuff(subject, recording_for_subject):
    assert(subject > 0)
    assert(recording_for_subject > 0)
    matlab_stuff = [[[1, 0, 0, 1, 25, 6416],
     [0, 1, 0, 1, 50, 12430]],
     [[1, 0, 0, 1, 25,  6502],
     [1, 1, 1, 1, 25,  6081]],
     [[1, 0, 0, 1, 50, 12488],
     [1, 1, 1, 1, 50, 12283]],
     [[1, 0, 0, 1, 25,  6171],
     [0, 1, 0, 1, 25,  6675]],
     [[1, 0, 0, 1, 50, 12820],
     [1, 1, 1, 1, 50, 12312]],
     [[1, 0, 0, 1, 25,  6188],
     [1, 1, 1, 1, 25,  6145]],
     [[1, 1, 1, 1, 25,  6239],
     [1, 0, 0, 1, 25,  6320]],
     [[1, 1, 1, 1, 25,  6468],
     [1, 0, 0, 1, 25,  6054]]]
     # bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, fps, num_frames = matlab_stuff[subject][recording]
    return matlab_stuff[subject-1][recording_for_subject-1]


class RawDatasetTrain():
    TRAINDIR = 'mpi_inf_3dhp'
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            dataset_dir = get_dataset_dir()
        train_dir = os.path.join(dataset_dir, MPIINFDIR, self.TRAINDIR)
        subjects = os.listdir(train_dir)
        subjects = sorted(list(map(lambda x: int(x[1:]), subjects)))
        subjects_ret = []
        print(subjects)
        for subject in subjects:
            subject_dir = os.path.join(train_dir, 'S{}'.format(subject))
            recs_for_subject = os.listdir(subject_dir)
            recs_for_subject = sorted(list(map(lambda x: int(x[3:]), recs_for_subject)))
            recs_ret = []
            print(recs_for_subject)
            for rec in recs_for_subject:
                seq = SingleTrainSequence(train_dir, subject, rec)
                recs_ret.append(seq)
            subjects_ret.append(recs_ret)
        self.subjects = subjects_ret

class SingleTrainSequence():
    def __init__(self, path, subj_idx, seq_idx):
        path = os.path.join(path, 'S{}'.format(subj_idx), 'Seq{}'.format(seq_idx))
        self.check_files(path)
        self.video_dir = os.path.join(path, 'imageSequence')
        self.fg_dir = os.path.join(path, 'FGmasks')
        self.chair_mask_dir = os.path.join(path, 'ChairMasks')
        self.calibration_data = self.parse_calibration_data(path)
        bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, fps, num_frames = get_matlab_stuff(subj_idx, seq_idx)
        self.pos2d, self.pos3d = self.parse_annotation_data(path, num_frames)
        self.fps = fps
        self.background_augmentable = bg_augmentable
        self.upper_body_augmentable = ub_augmentable
        self.lower_body_augmentable = lb_augmentable
        self.chair_augmentable = chair_augmentable

    @staticmethod
    def check_files(path):
        chair_mask_dir = os.path.join(path, 'ChairMasks')
        assert(os.path.exists(chair_mask_dir))
        assert(len(os.listdir(chair_mask_dir))==14)
        fg_mask_dir = os.path.join(path, 'FGmasks')
        assert(os.path.exists(fg_mask_dir))
        assert(len(os.listdir(fg_mask_dir))==14)
        image_seq_dir = os.path.join(path, 'imageSequence')
        assert(os.path.exists(image_seq_dir))
        assert(len(os.listdir(image_seq_dir))==14)
        annot_path = os.path.join(path, 'annot.mat')
        assert(os.path.exists(annot_path))
        cam_calib_path = os.path.join(path, 'camera.calibration')
        assert(os.path.exists(cam_calib_path))

    @staticmethod
    def parse_calibration_data(path):
        cam_calib_path = os.path.join(path, 'camera.calibration')
        with open(cam_calib_path, 'r') as f:
            lines = f.read()
        camera_data_list = lines.split('name')[1:] # first line is a data format
        ret_dict = {}
        for camera_data in camera_data_list:
            camera_fields = camera_data.split('\n')
            camera_id = int(camera_fields[0].strip())
            intrinsic_set = False
            extrinsic_set = False
            for field in camera_fields[1:]:
                field = field.strip().split()
                if len(field) == 0:
                    continue
                fieldname = field[0]
                field = field[1:]
                if fieldname == 'sensor':
                    assert(len(field) == 2)
                    assert(int(field[0]) == 10)
                    assert(int(field[1]) == 10)
                elif fieldname == 'size':
                    assert(len(field) == 2)
                    assert(int(field[0]) == 2048)
                    assert(int(field[1]) == 2048)
                elif fieldname == 'animated':
                    assert(len(field) == 1)
                    assert(int(field[0]) == 0)
                elif fieldname == 'intrinsic':
                    assert(len(field) == 16)
                    intrinsic = np.array(list(map(float, field))).reshape(4,4)
                    assert(np.sum(np.abs(intrinsic[:,3] - np.array([0.0,0,0,1]))) == 0)
                    assert(np.sum(np.abs(intrinsic[3,:] - np.array([0.0,0,0,1]))) == 0)
                    intrinsic = intrinsic[:3, :3]
                    assert(intrinsic_set == False)
                    intrinsic_set = True
                elif fieldname == 'extrinsic':
                    assert(len(field) == 16)
                    extrinsic = np.array(list(map(float, field))).reshape(4,4)
                    R = extrinsic[:3, :3]
                    t = extrinsic[:3, 3]
                    assert(np.sum(np.abs(extrinsic[3,:] - np.array([0.0,0,0,1]))) == 0)
                    assert(extrinsic_set == False)
                    extrinsic_set = True
                elif fieldname == 'radial':
                    assert(len(field) == 1)
                    assert(float(field[0]) == 0)
                else:
                    print('unknown fieldname {}'.format(fieldname))
                    exit(0)
            assert(extrinsic_set)
            assert(intrinsic_set)
            ret_dict[camera_id] = (intrinsic, R, t)
        retlist = []
        for i in range(14):
            retlist.append(ret_dict[i])
        return retlist

    @staticmethod
    def parse_annotation_data(path, max_frames):
        annot_path = os.path.join(path, 'annot.mat')
        mat_data = scipy.io.loadmat(annot_path)
        # poses3d = mat_data['annot3'].reshape(-1) # is relative to camera, use matmul(R.T, x-t) to go to global
        poses3d = mat_data['annot3'].reshape(-1) # is annot3 scaled to universal size
        poses2d = mat_data['annot2'].reshape(-1) # poses 2d is just a projection of poses3d, using the intrinsic.
        cameras = mat_data['cameras'].reshape(-1)
        frames = mat_data['frames'].reshape(-1)
        assert(np.all(frames == np.arange(len(frames))))
        assert(frames[0] == 0)
        pos2d = np.empty((14, max_frames, 28, 2), dtype = np.float64)
        pos3d = np.empty((14, max_frames, 28, 3), dtype = np.float64)
        assert(len(frames) >= max_frames)
        for i, c, p2, p3 in zip(range(len(cameras)), cameras, poses2d, poses3d):
            pos2d[c] = poses2d[i].reshape(-1, 28 ,2)[:max_frames]
            pos3d[c] = poses3d[i].reshape(-1, 28 ,3)[:max_frames]
        return pos2d, pos3d

    def load_videos(self, camera_idx):
        vidname = os.path.join(self.video_dir, 'video_{}.avi'.format(camera_idx))
        stream = cv2.VideoCapture(vidname)
        vidname_chair_mask = os.path.join(self.chair_mask_dir, 'video_{}.avi'.format(camera_idx))
        chair_mask = cv2.VideoCapture(vidname_chair_mask)
        vidname_fg_mask = os.path.join(self.fg_dir, 'video_{}.avi'.format(camera_idx))
        fg_masks = cv2.VideoCapture(vidname_fg_mask)
        return stream, chair_mask, fg_masks

def visualize_seek(single_recording_dataset, frame, cam=0):
    success_stream = True
    current_frame = -1
    stream, chair_mask_stream, fg_mask_stream = single_recording_dataset.load_videos(cam)
    _, _, _, _, fps, num_frames = get_matlab_stuff(1,1)
    frame_float = frame/num_frames

    while success_stream:
        success_stream, image = stream.read()
        current_frame += 1
        if frame == current_frame:
            break
    stream.release()
    chair_mask_stream.release()
    fg_mask_stream.release()
    stream, chair_mask_stream, fg_mask_stream = single_recording_dataset.load_videos(cam)

    print(stream)
    print(frame_float)
    stream.set(1, frame)
    print(stream)
    success_stream, image2 = stream.read()
    print(success_stream)
    stream.release()
    chair_mask_stream.release()
    fg_mask_stream.release()
    plt.imshow(image)
    plt.show()
    print(image2)
    plt.imshow(image2)
    plt.show()
    plt.imshow((image-image2)*1.0)
    plt.show()



def visualize_raw_annotation_data(single_recording_dataset, vis_frequency=None):
    # poses3d_univ = mat_data['univ_annot3'].reshape(-1) univ is junk, the skeleton is resized to get a constant length.

    for i in range(single_recording_dataset.pos2d.shape[0]):
        I, R, t = single_recording_dataset.calibration_data[i]
        pos2d = single_recording_dataset.pos2d[i]
        pos3d = single_recording_dataset.pos3d[i]
        stream, chair_mask_stream, fg_mask_stream = single_recording_dataset.load_videos(i)
        frame = -1
        spf = 1/single_recording_dataset.fps
        next_vis_time = 0
        current_time = 0
        success_stream = True
        success_chair_mask = True
        success_fg_mask = True
        if vis_frequency is None:
            vis_frequency = 0.0
        while(success_stream and success_chair_mask and success_fg_mask):
            if frame > pos3d.shape[0]:
                break
            success_stream, image = stream.read()
            success_fg_mask, fg_mask = fg_mask_stream.read()
            success_chair_mask, chair_mask = chair_mask_stream.read()
            frame += 1
            if current_time < next_vis_time:
                current_time += spf
                continue
            current_time += spf
            next_vis_time = next_vis_time + vis_frequency
            image = image[:,:,::-1]/255.0
            bbx_max = np.max(pos2d[frame], axis=0)
            bbx_min = np.min(pos2d[frame], axis=0)
            center = (bbx_max+bbx_min)/2
            size = bbx_max-bbx_min
            size = np.max(size)*1.2
            plt.plot(pos2d[frame, :,0], pos2d[frame, :,1], 'x')
            bbx_start_x = center[0] - size/2
            bbx_start_y = center[1] - size/2
            bbx_end_x = center[0] + size/2
            bbx_end_y = center[1] + size/2
            plt.plot(pos2d[frame, :,0], pos2d[frame, :,1], 'x')
            plt.plot([bbx_start_x, bbx_end_x, bbx_end_x, bbx_start_x, bbx_start_x], [bbx_start_y, bbx_start_y, bbx_end_y, bbx_end_y, bbx_start_y], 'r')
            greenscreen_remover = (fg_mask[:,:,2] > 128).reshape(fg_mask.shape[0], fg_mask.shape[1], 1)
            magenta = np.array([1.0, 0, 1.0]).reshape(1,1,3)
            s = greenscreen_remover.shape
            magenta = magenta * np.logical_not(greenscreen_remover).reshape(s[0], s[1], 1)
            plt.imshow(image*greenscreen_remover+magenta)
            plt.show()
            plt.imshow(image)
            plt.show()

def plot_bbx(bbx_start_x, bbx_end_x, bbx_start_y, bbx_end_y, c='r'):
    plt.plot([bbx_start_x, bbx_end_x, bbx_end_x, bbx_start_x, bbx_start_x], [bbx_start_y, bbx_start_y, bbx_end_y, bbx_end_y, bbx_start_y], c)


def export_camera_sequence(streams, avi_path, calibration_data, pos2d, pos3d, dump_dir, matlab_stuff):
    def save_calib_data(dump_dir, calibration_data):
        cam_calib_savepath = os.path.join(dump_dir, 'calib.json')
        I, R, t = calibration_data
        I_json = [list(I[0]), list(I[1]), list(I[2])]
        R_json = [list(R[0]), list(R[1]), list(R[2])]
        t_json = list(t)
        cam_json = {'R':R_json, 'I':I_json, 't':t_json}
        with open(cam_calib_savepath, 'w') as f:
            json.dump(cam_json, f)

    def save_masks(dump_dir, streams, matlab_stuff, max_frame):
        frame = -1
        bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, fps, num_frames = matlab_stuff
        success_chair_mask = True
        success_fg_mask = True

        stream, chair_mask_stream, fg_mask_stream = streams
        stream.release # only preprocess masks
        mask_dir = os.path.join(dump_dir, 'mask')
        os.makedirs(mask_dir)
        while(success_chair_mask and success_fg_mask):
            frame += 1
            if frame > max_frame:
                break
            success_fg_mask, fg_mask = fg_mask_stream.read()
            success_chair_mask, chair_mask = chair_mask_stream.read()
            if not(success_fg_mask and success_chair_mask):
                print('failed at frame {}'.format(frame))
                break
            fg_mask = fg_mask[:,:,::-1]
            chair_mask = chair_mask[:,:,::-1]

            mask, augmentable = extract_idx_mask(chair_mask, fg_mask, bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable)
            saveim = Image.fromarray(mask)
            savepath = os.path.join(mask_dir, '{}.png'.format(frame))
            saveim.save(savepath)
        return augmentable

    def save_frame_data(dump_dir, augmentable, pos2d, pos3d, max_frame):
        frame_data_dir = os.path.join(dump_dir, 'frame_data')
        os.makedirs(frame_data_dir)
        for frame in range(max_frame):
            pos2d_frame = pos2d[frame]
            pos3d_frame = pos3d[frame]
            bbx_max = np.max(pos2d_frame, axis=0)
            bbx_min = np.min(pos2d_frame, axis=0)
            bbx_ideal = [int(bbx_min[0]), int(bbx_max[0]), int(bbx_min[1]), int(bbx_max[1])]
            center = (bbx_max+bbx_min)/2
            size = bbx_max-bbx_min
            size_ideal = np.max(size)*1.2
            size_noise = size_ideal * (0.95+0.1*np.random.uniform())
            center_noise = center + size_ideal * (2*np.random.uniform(size=(2))-1)*0.05
            bbx_start_x = center_noise[0] - size_noise/2
            bbx_end_x = center_noise[0] + size_noise/2
            bbx_start_y = center_noise[1] - size_noise/2
            bbx_end_y = center_noise[1] + size_noise/2
   
            pos3d_frame = list(map(list, pos3d_frame))
            pos2d_frame = list(map(list, pos2d_frame))
            bbx_noise = [int(bbx_start_x), int(bbx_end_x), int(bbx_start_y), int(bbx_end_y)]
            ann = {'bbx_ideal': bbx_ideal, 'bbx_noise': bbx_noise, 'pos2d':pos2d_frame, 'pos3d':pos3d_frame, 'augmentable': augmentable}
            annotation_save_path = os.path.join(frame_data_dir, '{}.json'.format(frame))
            with open(annotation_save_path, 'w') as f:
                json.dump(ann, f)

    def move_video_file(dump_dir, avi_path):
        video_path_out = os.path.join(dump_dir, 'vid.avi')
        shutil.copyfile(avi_path, video_path_out)

    # execute defined functions
    bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, fps, num_frames = matlab_stuff
    print('num_frames x3')
    print(num_frames, pos3d.shape[0], pos2d.shape[0])
    max_frame = min(num_frames, pos3d.shape[0], pos2d.shape[0])
    save_calib_data(dump_dir, calibration_data)
    augmentable = save_masks(dump_dir, streams, matlab_stuff, max_frame)
    save_frame_data(dump_dir, augmentable, pos2d, pos3d, max_frame)
    move_video_file(dump_dir, avi_path)




def extract_idx_mask(chair_mask, fg_mask, bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable):
    fg_mask = fg_mask.astype(np.float)/255.0
    chair_mask = chair_mask.astype(np.float)/255.0
    if chair_augmentable:
        score_chair = 2-2*chair_mask[:,:,0]
    else:
        score_chair = np.zeros((chair_mask.shape[0], chair_mask.shape[1]))
    if ub_augmentable or lb_augmentable:
        # you are now in ymc space
        m = np.array([1.0,0,1]).reshape(1,1,3) / np.sqrt(2)
        y = np.array([1.0,1,0]).reshape(1,1,3) / np.sqrt(2)
        c = np.array([0.0,1,1]).reshape(1,1,3) / np.sqrt(2)
        w = np.array([1.0,1,1]).reshape(1,1,3) / np.sqrt(3)
        score_bg = np.sum(fg_mask * c, axis=2)
        score_base = np.sum(fg_mask * w, axis=2)
        if ub_augmentable:
            score_ub = np.sum(fg_mask * m, axis=2)
        else:
            score_ub = np.zeros((chair_mask.shape[0], chair_mask.shape[1]))
        if lb_augmentable:
            score_lb = np.sum(fg_mask * y, axis=2)
        else:
            score_lb = np.zeros((chair_mask.shape[0], chair_mask.shape[1]))
    else:
        score_base = np.ones((chair_mask.shape[0], chair_mask.shape[1]))
        # red-black space
        score_bg = 2-2*fg_mask[:,:,0]
        score_ub = np.zeros((chair_mask.shape[0], chair_mask.shape[1]))
        score_lb = np.zeros((chair_mask.shape[0], chair_mask.shape[1]))
    v = np.argmax(np.stack([score_base, score_bg, score_ub, score_lb, score_chair]), axis=0)
    v = v.astype(np.uint8)
    bg_augmentable = bg_augmentable and np.any(v==1)
    ub_augmentable = ub_augmentable and np.any(v==2)
    lb_augmentable = lb_augmentable and np.any(v==3)
    chair_augmentable = chair_augmentable and np.any(v==4)
    augmentable = [bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable]
    augmentable = list(map(int, augmentable))

    return v, augmentable



MPIINF_PREPROCESSED_DIR = 'MPIINF_preprocessed'
def export_raw_annotation_data(raw_dataset, dataset_dir=None, target_subjects=None):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    save_dir = os.path.join(dataset_dir, MPIINF_PREPROCESSED_DIR)
    if target_subjects is None:
        target_subjects = list(np.arange((len(raw_dataset.subjects)))+1)
    for subject_idx in target_subjects:
        # subject_idx is 0 indexed
        subject_data = raw_dataset.subjects[subject_idx]
        for seq_idx in range(len(subject_data)):
            # seq_idx is 0 indexed
            print('seq_idx: {}'.format(seq_idx))
            sequence = subject_data[seq_idx]
            for cam_idx in range(sequence.pos2d.shape[0]):
                print('cam_idx: {}'.format(cam_idx))
                calibration_data = sequence.calibration_data[cam_idx]
                pos2d = sequence.pos2d[cam_idx]
                pos3d = sequence.pos3d[cam_idx]
                dump_dir = os.path.join(save_dir, str(subject_idx), str(seq_idx), str(cam_idx))
                if os.path.exists(dump_dir):
                    shutil.rmtree(dump_dir)
                os.makedirs(dump_dir)
                matlab_stuff = get_matlab_stuff(subject_idx+1, seq_idx+1) # this function takes 1 indexed things
                streams = sequence.load_videos(cam_idx)
                video_file = os.path.join(sequence.video_dir, 'video_{}.avi'.format(cam_idx))
                export_camera_sequence(streams, video_file, calibration_data, pos2d, pos3d, dump_dir, matlab_stuff)


def main_preprocess_dataset():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--target_subjects', type=int, default=-1)
    args = arg_parser.parse_args()
    target_subjects = args.target_subjects
    if target_subjects == -1:
        target_subjects = None
    else:
        target_subjects = [target_subjects]
    raw_dataset = RawDatasetTrain()
    export_raw_annotation_data(raw_dataset, target_subjects=target_subjects)

def main():
    # note the function below parses input args
    main_preprocess_dataset()

if __name__ == '__main__':
    main()
