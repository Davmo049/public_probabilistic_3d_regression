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

def create_resnet(dim_out, device='cpu'):
    group_norm_f = lambda x: torch.nn.GroupNorm(1,x)
    net = resnet.resnet50(num_classes=dim_out, norm_layer=group_norm_f)
    with torch.no_grad():
        net.fc.bias.fill_(0)
        net.fc.weight.fill_(0)
    net = net.to(device)
    return net

def get_loss(mu, D, S, device, prob=True, squared=False, euclidean=False, logz=False):
    if prob:
        assert(squared == False)
        assert(euclidean == False)
        assert(logz == False)
        name = 'Prob'
        return Losses.Loss3d(mu, S, device, torch.float32), name
    if euclidean:
        assert(not(logz))
        std = torch.tensor((0.25,0.25,1))
        mean = torch.tensor((0,0,4))
        name = 'Euc'
    else:
        std = torch.tensor((1,1,D))
        mean = torch.tensor((0,0,mu))
        name = 'Norm'
        if logz:
            name += 'Logz'
    if squared:
        name += 'Square'
    return Losses.BaselineLoss(mean,std,device, S,squared,euclidean,logz), name

def train_epoch(net, loss_function, opt, logger, dataloader, device, use_incorrect_focal_length=False):
    net.eval()
    for image, f, v, annotated, valid_depth in tqdm.tqdm(dataloader):
        image=image.to(device)
        f = f.to(device, torch.float32)
        if use_incorrect_focal_length:
            with torch.no_grad():
                f.fill_(np.sqrt(2000*1200))
        v = v.to(device)
        valid_depth = valid_depth.to(device)
        annotated = annotated.to(device)
        pred = net(image).view(v.shape[0], v.shape[1], -1)
        losses, A, a_z, mu_v, mu_z = loss_function(v, f, pred, annotated, valid_depth)
        loss = torch.mean(losses.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        # log
        losses = losses.detach().cpu().numpy()
        mu_v = mu_v.detach().cpu().numpy()
        mu_z = mu_z.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        annotated = annotated.cpu().numpy()
        valid_depth = valid_depth.cpu().numpy()
        a_z = a_z.detach().cpu().numpy()
        A = A.detach().cpu().numpy()
        logger.add(losses, mu_v, mu_z, v, annotated, valid_depth, a_z, A)
        del image, f, v, annotated, valid_depth, pred, losses, A, a_z, mu_v, mu_z, loss
    logger.finish()

def eval_epoch(net, loss_function, logger, dataloader, device, use_incorrect_focal_length=False):
    net.eval()
    with torch.no_grad():
        for image, f, v, ann, valid_depth in dataloader: # all from 3d dataset all kps annotated
            image=image.to(device)
            if use_incorrect_focal_length:
                f.fill_(np.sqrt(2000*1200))
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
            logger.add(losses, mu_v, mu_z, v, ann, valid_depth, a_z, A)
    logger.finish()



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


def main():
    # subset = 'large'
    # subset = 'default'
    # subset = 'scale_variation'
    # subset = 'no_occlusion_scale_variation'
    subset = 'no_occlusion'
    batch_size = 64
    path = os.path.join(get_dataset_dir(), Dataset.SYNTHETIC_DATA_PATH, subset)
    dl_train, dl_eval, muz0D = get_dataloaders(path, batch_size)
    output_size=224
    device = torch.device('cuda')
    loss_prob = True
    loss_squared = False
    loss_euclidean = False
    loss_logz = False
    use_incorrect_focal_length = False
    loss_function, lossname = get_loss(muz0D[0], muz0D[1], output_size, device, prob=loss_prob, squared=loss_squared, euclidean=loss_euclidean, logz=loss_logz)
    train_name = 'slice_{}_bs_{}_{}'.format(subset, batch_size, lossname)
    if use_incorrect_focal_length:
        train_name += '_wrong_f'
    print(train_name)

    if loss_prob:
        net_dim_out = Losses.END_DIST_PARAM
    else:
        net_dim_out = 3
    net = create_resnet(net_dim_out, device=device)
    opt = torch.optim.Adam(net.parameters())
    epochs = 50
    logger = logger_lib.Logger(train_name, load=False)

    for e in range(epochs):
        logger.set_epoch(e)
        logger_train = logger.get_train()
        train_epoch(net, loss_function, opt, logger_train, dl_train, device, use_incorrect_focal_length)
        if e == (epochs-1):
            logger.save_network(net)
        logger_eval = logger.get_eval()
        eval_epoch(net, loss_function, logger_eval, dl_eval, device, use_incorrect_focal_length)
    logger.finish_training()


if __name__ == '__main__':
    main()
