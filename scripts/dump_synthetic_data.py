import os

import numpy as np
import torch
import torchvision.models.resnet as resnet
import torchvision
import tqdm

import Datasets.SytheticDataset.Dataset as Dataset
from general_utils.environment_variables import get_dataset_dir
import logger as logger_lib
import Losses


def get_dataloaders(path, batch_size):
    dataset = Dataset.Dataset(path)
    train_ds = dataset.get_train()
    dataloader_train = torch.utils.data.DataLoader(train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)

    eval_ds = dataset.get_eval()
    dataloader_eval = torch.utils.data.DataLoader(eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval, (train_ds.mu0, train_ds.D)

def serialize_vecs(vs):
    s = '['
    vstrings = []
    for v in vs:
        vstrings.append('['+' '.join(list(map(str, v)))+']')
    return '['+','.join(vstrings)+']'


def serialize_mats(As):
    return serialize_vecs(list(map(lambda x: x.flatten(), As)))


def deserialize_vecs(s):
    s = s[1:-1]
    vstrings = s.split(',')
    vs = []
    for vstring in vstrings:
        vstring = vstring[1:-1]
        v = np.array(list(map(float, vstring.split(' '))))
        vs.append(v)
    return vs


def deserialize_mats(s):
    v = deserialize_vecs(s)
    return np.array(list(map(lambda x: x.reshape(2,2), v)))


def eval_epoch(net, loss_function, dataloader, device, dumpname):
    net.eval()
    As = []
    azs = []
    mu_vs = []
    mu_zs = []
    vs = []
    fs = []
    with torch.no_grad():
        for image, f, v, ann, valid_depth in dataloader: # all from 3d dataset all kps annotated
            image=image.to(device)
            f = f.to(device, torch.float32)
            v = v.to(device)
            ann = ann.to(device)
            valid_depth = valid_depth.to(device)
            pred = net(image).view(v.shape[0], v.shape[1], -1)
            losses, A, a_z, mu_v, mu_z = loss_function(v, f, pred, ann, valid_depth)

            losses = losses.detach().cpu().numpy()
            mu_v = mu_v.detach().cpu().numpy()
            mu_z = mu_z.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
            ann = ann.cpu().numpy()
            valid_depth = valid_depth.cpu().numpy()
            a_z = a_z.detach().cpu().numpy()
            A = A.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            for i in range(f.shape[0]):
                As.append(A[i,0])
                azs.append(a_z[i,0])
                mu_vs.append(mu_v[i,0])
                mu_zs.append(mu_z[i,0,0])
                vs.append(v[i,0])
                fs.append(f[i])
    s = 'A:' + serialize_mats(As) +'\n'
    s += 'az:' + str(azs) +'\n'
    s += 'mu_v:' + serialize_vecs(mu_vs) +'\n'
    s += 'mu_z:' + str(mu_zs) +'\n'
    s += 'v:' + serialize_vecs(vs) +'\n'
    s += 'f:' + str(fs) +'\n'
    print(dumpname)
    with open(dumpname, 'w') as f:
        f.write(s)

def main():
    subset = []
    subset.append('default')
    subset.append('scale_variation')
    subset.append('no_occlusion_scale_variation')
    for s in subset:
        main_dump(s)

def main_dump(subset):
    # subset = 'no_occlusion'
    batch_size = 64
    output_size=224
    device = torch.device('cuda')
    path = os.path.join(get_dataset_dir(), Dataset.SYNTHETIC_DATA_PATH, subset)
    dl_train, dl_eval, muz0D = get_dataloaders(path, batch_size)
    loss_function, lossname = Losses.Loss3d(muz0D[0], output_size, device, torch.float32), 'Prob'

    load_name = 'slice_{}_bs_{}_{}'.format(subset, batch_size, lossname)
    net_dim_out = Losses.END_DIST_PARAM

    group_norm_f = lambda x: torch.nn.GroupNorm(1,x)
    net = resnet.resnet50(num_classes=net_dim_out, norm_layer=group_norm_f)
    net = net.to(device)

    logger = logger_lib.Logger(load_name, load=True)
    logger.load_network_weights(49, net, device)
    eval_epoch(net, loss_function, dl_eval, device, dumpname='dump'+subset+'.txt')

if __name__ == '__main__':
    main()
