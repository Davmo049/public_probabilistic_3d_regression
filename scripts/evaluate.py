# import numpy as np
# import torch
# import torchvision.models.resnet as resnet
# 
import numpy as np
import torch
import torchvision.models.resnet as resnet

from Datasets.MuPoTS.PreprocessedDataset import PreprocessedDataset
import Datasets.mpiinf.PreprocDataset as PreprocDataset
import Datasets.mpiinf.PreprocDataset as mpii3dhp
import Datasets.PW3D.PreprocessedDataset as PW3D
import Datasets.MuPoTS.PreprocessedDataset as MuPoTS
import logger as logger_lib
import Losses


def main():
    device = torch.device('cuda')
    # choose dataset
    dataset_name = 'mupots'
    print(dataset_name)
    if dataset_name == 'mupots':
        dataset = MuPoTS.PreprocessedDataset()
    elif dataset_name == 'pw3d':
        validate_pw3d=False
        _, dataset = PW3D.get_normal(validate=validate_pw3d)
    elif dataset_name == 'mpiinf':
        dataset = PreprocDataset.PreprocessedDataset(augment=False, load_from_crop_frame_dump=True)
    else:
        print('unknown dataset')
        exit(0)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    # network create
    num_keypoints_per_sample = 28
    net = resnet.resnet50(num_classes=Losses.END_DIST_PARAM*num_keypoints_per_sample)
    net = net.to(device)
    # network load
    # train_name = 'updated_loss'
    train_name = 'train_after_TMLR'
    load_epoch=49
    logger = logger_lib.Logger(train_name, load=True)
    logger.load_network_weights(load_epoch, net, device, transfer=False)
    # create loss remapper
    loss_function = Losses.Loss3d(mpii3dhp.muz0, dataset.output_size, device)
    all_errors = []
    mean_err = []
    pck = []
    N = 0
    depth = []
    print_logger = logger_lib.Logger('', log_to_file = False)
    print_logger_eval = print_logger.get_eval()
    net.eval()
    for image, f, v, annotated, valid_depth in dataloader:
        image = image.to(device)
        f = f.to(device, torch.float32)
        pred = net(image).view(f.shape[0], num_keypoints_per_sample, Losses.END_DIST_PARAM)
        A, a_z, mu_v, mu_z = loss_function.only_remap(f, pred)
        mu_v = mu_v.detach().cpu().numpy()
        mu_z = mu_z.detach().cpu().numpy()
        v = v.detach().numpy()
        B = image.shape[0]
        annotated = annotated.detach().cpu().numpy()
        valid_depth = valid_depth.detach().cpu().numpy()
        losses = np.zeros(f.shape[0])
        a_z = a_z.detach().cpu().numpy()
        A = A.detach().cpu().numpy()
        print_logger_eval.add(losses, mu_v, mu_z, v, annotated, valid_depth, a_z, A)
    print_logger_eval.finish()

if __name__ == '__main__':
    main()
