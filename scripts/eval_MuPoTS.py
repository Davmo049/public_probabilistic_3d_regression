# import numpy as np
# import torch
# import torchvision.models.resnet as resnet
# 
import numpy as np
import torch
import torchvision.models.resnet as resnet

from Datasets.MuPoTS.PreprocessedDataset import PreprocessedDataset
import Datasets.mpiinf.PreprocDataset as mpii3dhp
import logger as logger_lib
import Losses

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')


def main():
    device = torch.device('cpu')
    dataset = PreprocessedDataset()
    # ds_eval = MPII2d.Dataset()
    # ds_eval = ds_train
    dataloader_eval = torch.utils.data.DataLoader(dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    logger = logger_lib.Logger('try_to_get_two_datasets_to_work', load=True)
    loss_function = Losses.Loss3d(mpii3dhp.muz0, dataset.output_size, device)
    num_keypoints_per_sample = 28
    net = resnet.resnet50(num_classes=Losses.END_DIST_PARAM*num_keypoints_per_sample)
    load_epoch=179
    logger.load_network_weights(load_epoch, net, device, transfer=False)
    net.eval()
    all_to_ann = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]
    all_errors = []
    mean_err = []
    pck = 0
    N = 0
    depth = []
    for image, f, v, annotated, valid_depth, ideal_to_real in dataloader_eval:
        pred = net(image).view(f.shape[0], num_keypoints_per_sample, Losses.END_DIST_PARAM)
        A, a_z, mu_v, mu_z = loss_function.only_remap(f, pred)
        mu_v = mu_v.detach().cpu().numpy()
        mu_z = mu_z.detach().cpu().numpy()
        ideal_to_real_np = ideal_to_real.numpy()[0]
        x = mu_v[0,:,0].reshape(-1)
        y = mu_v[0,:,1].reshape(-1)
        z = mu_z[0].reshape(-1)
        x = x*z
        y = y*z
        v = v[0].numpy()
        v_est = np.stack([x,y,z], axis=1)
        v_est = v_est[all_to_ann]
        if True:
            v_est = v_est.transpose()
            v = np.matmul(np.linalg.inv(ideal_to_real_np[:3,:3]), v)
        else:
            v_est = np.matmul(ideal_to_real_np[:3,:3], v_est.transpose())
        # relative, if desired
        # v_est += (v[:,14]-v_est[:,14]).reshape(3,1)
        v_est_filt = v_est[:,:-2]
        v_filt = v[:,:-2]
        error_joints = np.linalg.norm(v_filt-v_est_filt,axis=0)
        depth.append(v[2,14])
        mean_err.append(np.mean(error_joints))
        for e in error_joints:
            all_errors.append(e)
        pck += np.sum(error_joints < 250)
        N += len(error_joints)
        if True:
            x = v_est[0]
            y = v_est[1]
            z = v_est[2]
            f = f[0].numpy()
            image_np = image[0].numpy().transpose(1,2,0)
            print(all_errors[-1])
            draw_image_skeleton(image_np, f, x, y, z, v.transpose())
            vis_3d_skeleton(x,y,z, v.transpose())
        if len(all_errors) > 1700:
            break
    print(pck/N)
    plt.hist(all_errors)
    plt.show()
    print(np.mean(depth))
    plt.plot(depth, mean_err, 'rx')
    plt.show()
    print(np.mean(all_errors))
    print(np.mean(sorted(all_errors)[:int(len(all_errors)*0.9)]))

def vis_3d_skeleton(x,y,z,v):
    x_gt = v[:,0]
    y_gt = v[:,1]
    z_gt = v[:,2]
    fig = plt.figure()
    x_range = (min(np.min(x), np.min(v[:,0])), max(np.max(x), np.max(v[:,0])))
    y_range = (min(np.min(y), np.min(v[:,1])), max(np.max(y), np.max(v[:,1])))
    z_range = (min(np.min(z), np.min(v[:,2])), max(np.max(z), np.max(v[:,2])))
    sz = max(x_range[1]-x_range[0], z_range[1]-z_range[0], z_range[1]-z_range[0])
    mid_x = np.mean(x_range)
    mid_y = np.mean(y_range)
    mid_z = np.mean(z_range)

    ax = fig.add_subplot(projection='3d')
    draw_ann3d(ax,x,y,z,'')
    draw_ann3d(ax,x_gt,y_gt,z_gt,'--')
    ax.scatter(0,0,0,'x')
    ax.axes.set_xlim3d(left=mid_x-sz/2, right=mid_x+sz/2)
    ax.axes.set_ylim3d(bottom=mid_z-sz/2, top=mid_z+sz/2)
    ax.axes.set_zlim3d(bottom=mid_y-sz/2, top=mid_y+sz/2)
    plt.show()

def plot_line_from_v(ax, v, c):
    ax.plot(v[:, 0], v[:, 2], -v[:,1], c)

def draw_image_skeleton(image, f, x, y, z, gt):
    plt.imshow(image)
    xp = f*x/z+112
    yp = f*y/z+112
    xp_gt = f*gt[:,0]/gt[:,2]+112
    yp_gt = f*gt[:,1]/gt[:,2]+112
    draw_ann(xp, yp, '')
    draw_ann(xp_gt, yp_gt, '--')
    plt.show()

def draw_ann(x,y, modifier):
    for l, color in zip(leaves, colors):
        cur = l
        draw = []
        draw.append(cur)
        while True:
            cur = parent[cur]
            draw.append(cur)
            if cur in stops:
                break
        plt.plot(x[draw], y[draw], color+modifier)

def draw_ann3d(ax,x,y,z,modifier):
    for l, color in zip(leaves, colors):
        cur = l
        draw = []
        draw.append(cur)
        while True:
            cur = parent[cur]
            draw.append(cur)
            if cur in stops:
                break
        ax.plot(x[draw], z[draw], -y[draw], color+modifier)

leaves = [0, 1,4,7,10,13]#,9,10,13,16]
stops = set([1,14])
colors = ['g','g','r','b','r','b']
parent = [16,15,1,2,3,1,5,6,14,8,9,14,11,12,14,14,1]

if __name__ == '__main__':
    main()
