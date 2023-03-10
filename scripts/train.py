import Datasets.mpiinf.PreprocDataset as PreprocDataset
import torchvision.models.resnet as resnet
import torch
import Losses
import logger as logger_lib
import numpy as np
import Datasets.mpii_pose.PreprocessedDataset as MPII2d
import Datasets.Coco.PreprocessedDataset as Coco
import Datasets.PW3D.PreprocessedDataset as PW3D
import tqdm

# profile

def get_loss(mu, D, S, device, prob=True, logz=False):
    if prob:
        assert(logz == False)
        name = 'Prob'
        return Losses.Loss3d(mu, S, device, torch.float32), name
    else:
        std = torch.tensor((1,1,D))
        mean = torch.tensor((0,0,mu))
        name = 'Norm'
        if logz:
            name += 'Logz'
    return Losses.BaselineLoss(mean,std,device, S,squared,euclidean,logz), name


class ThinDataset():
    def __init__(self, dataset, start_idx, end_idx=None):
        if isinstance(start_idx, list):
            assert(end_idx is None)
            self.samples = start_idx
        else:
            assert(start_idx >= 0)
            assert(end_idx <= len(dataset))
            assert(start_idx < end_idx)
            self.samples = list(range(start_idx, end_idx))
        self.ds = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.ds[self.samples[i]]

def large_grad_hook(x, v,f,pred,ann, valid_depth):
    if torch.any(torch.isnan(x)):
        for i in range(x.shape[0]):
            val = x[i]
            if torch.any(torch.isnan(val)):
                for j in range(x.shape[1]):
                    kp = x[i, j]
                    if torch.any(torch.isnan(kp)):
                        print('x')
                        print(x[i,j])
                        print('v')
                        print(v[i,j])
                        print('f')
                        print(f)
                        print('ann')
                        print(ann[i,j])
                        print('valdep')
                        print(valid_depth[i])
                        exit(0)
    if torch.sum(torch.abs(x.view(-1))) > 10:
        maxv = -1
        maxindex = -1
        for i in range(x.shape[0]):
            cand = torch.sum(torch.abs(x[i].view(-1)))
            if cand > maxv:
                maxv = cand
                maxindex = i
        print('x')
        print(x[maxindex])
        print('f')
        print(f[maxindex])
        print('v')
        print(v[maxindex])
        print('pred')
        print(pred[maxindex])
        print('ann')
        print(ann[maxindex])
        print('valid_depth')
        print(valid_depth[maxindex])
        exit(0)

def train_epoch(net, loss_function, opt, logger, dataloader, device, output_dim_per_kp):
    net.train()
    for image, f, v, annotated, valid_depth in tqdm.tqdm(dataloader):
        image=image.to(device)
        f = f.to(device, torch.float32)
        v = v.to(device)
        valid_depth = valid_depth.to(device)
        annotated = annotated.to(device)
        pred = net(image).view(v.shape[0], v.shape[1], output_dim_per_kp)
        pred.register_hook(lambda x: large_grad_hook(x, v,f,pred,annotated, valid_depth))
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
        logger.add(losses, mu_v, mu_z, v, annotated, valid_depth,a_z,A)
    logger.finish()

def eval_epoch(net, loss_function, logger, dataloader, device, output_dim_per_kp):
    net.eval()
    with torch.no_grad():
        for image, f, v, ann, valid_depth in dataloader: # all from 3d dataset all kps annotated
            image=image.to(device)
            f = f.to(device, torch.float32)
            v = v.to(device)
            ann = ann.to(device)
            valid_depth = valid_depth.to(device)
            pred = net(image).view(v.shape[0], v.shape[1], output_dim_per_kp)
            losses, A, a_z, mu_v, mu_z = loss_function(v, f, pred, ann, valid_depth)

            losses = losses.detach().cpu().numpy()
            mu_v = mu_v.detach().cpu().numpy()
            mu_z = mu_z.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
            ann = ann.cpu().numpy()
            valid_depth = valid_depth.cpu().numpy()
            a_z = a_z.detach().cpu().numpy()
            A = A.detach().cpu().numpy()
            logger.add(losses, mu_v, mu_z, v, ann, valid_depth,a_z,A)
    logger.finish()

def main():
    device = torch.device('cuda')
    output_size = 224
    train_on_normalized_data = False
    batch_size=32
    val_on_normalized_data = train_on_normalized_data
    dataset_mpiinf = PreprocDataset.PreprocessedDataset(augment=False, load_from_crop_frame_dump=True, output_size=output_size, normalize_height=train_on_normalized_data)
    ds_mpii2d = MPII2d.Dataset(output_size=output_size)
    ds_coco = Coco.get_normal()
    pw3d_train, pw3d_eval = PW3D.get_normal(normalize_train=train_on_normalized_data, normalize_val=val_on_normalized_data)
    dataloader_eval = torch.utils.data.DataLoader(pw3d_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)

    loss_prob = True
    loss_logz=False
    loss_function, loss_name = get_loss(PreprocDataset.muz0, PreprocDataset.D, output_size, device, prob=loss_prob, logz=loss_logz)
    output_dim_per_kp = Losses.END_DIST_PARAM
    if not(loss_prob):
        output_dim_per_kp = 3
    num_keypoints_per_sample = 28
    net = resnet.resnet50(num_classes=output_dim_per_kp*num_keypoints_per_sample)
    net = net.to(device)
    w_scale_factor = 1
    with torch.no_grad():
        net.fc.bias.fill_(0)
        net.fc.weight.fill_(0)
    opt = torch.optim.Adam(net.parameters())
    num_epochs = 50

    train_name = 'full_training' + '_' + loss_name
    load_from_epoch = -1
    logger = logger_lib.Logger(train_name, load=load_from_epoch!=-1)
    if load_from_epoch != -1:
        logger.load_network_weights(load_from_epoch, net, device)
    c = 0
    for e in range(load_from_epoch+1, num_epochs):
        ds_train = ThinDataset(dataset_mpiinf, list(range(c, len(dataset_mpiinf),9)))
        c = (c+4)%9
        # 9 cycles until repeat, decent spacing. gives approx double 3d to 2d data
        ds_train = torch.utils.data.ConcatDataset((ds_train, ds_mpii2d, ds_coco, pw3d_train))
        dataloader_train = torch.utils.data.DataLoader(ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
            pin_memory=True,
            drop_last=True)

        logger.set_epoch(e)
        logger_train = logger.get_train()
        train_epoch(net, loss_function, opt, logger_train, dataloader_train, device, output_dim_per_kp=output_dim_per_kp)

        if e == num_epochs-1:
            logger.save_network(net)

        logger_eval = logger.get_eval()
        eval_epoch(net, loss_function, logger_eval, dataloader_eval, device, output_dim_per_kp=output_dim_per_kp)
    logger.finish_training()

if __name__ == '__main__':
    main()
