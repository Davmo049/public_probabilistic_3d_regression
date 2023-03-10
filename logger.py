import numpy as np
import os
import tensorboardX
import torch

class WrappedTbWriter():
    def __init__(self, tb_writer):
        self.trainA = []
        self.evalA = []
        self.train_az = []
        self.eval_az = []
        self.tb_writer = tb_writer

    def add_scalar(self,s, v, epoch):
        if s == 'train/average_az':
            self.train_az.append(v)
        elif s == 'eval/average_az':
            self.eval_az.append(v)
        elif s == 'train/average_A':
            self.trainA.append(v)
        elif s == 'eval/average_A':
            self.evalA.append(v)
        self.tb_writer.add_scalar(s,v,epoch)

    def to_str(self):
        s = 'trainA:'+str(self.trainA) + '\n'
        s += 'evalA:'+str(self.evalA) + '\n'
        s += 'train_az:'+str(self.train_az) + '\n'
        s += 'eval_az:'+str(self.eval_az)
        return s


class Logger():
    def __init__(self, logger_path, load=False, log_to_file=True):
        self.epoch = 0
        self.logger_path = os.path.join('logs', logger_path)
        if not load and os.path.exists(self.logger_path) and log_to_file:
            print('{} exists'.format(self.logger_path))
            raise Exception("rerunning old training")
        if not(load) and log_to_file:
            os.makedirs(self.logger_path)
            os.makedirs(os.path.join(self.logger_path, 'saved_weights'))
        if log_to_file:
            tb_writer = tensorboardX.SummaryWriter(os.path.join(self.logger_path, 'tf_logging'))
            self.tb_writer = WrappedTbWriter(tb_writer)
        else:
            self.tb_writer = None

    def finish_training(self):
        with open(os.path.join(self.logger_path, 'saved_values.txt'), 'w') as f:
            f.write(self.tb_writer.to_str())

    def set_epoch(self, e):
        self.epoch = e

    def get_train(self):
        return LoggerInstance('train', self.epoch, self.tb_writer)

    def get_eval(self):
        return LoggerInstance('eval', self.epoch, self.tb_writer)

    def save_network(self, model):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(self.epoch))
        torch.save(model.state_dict(), path)

    def load_network_weights(self, epoch, model, device, transfer=False):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        if transfer:
            for v in ['head.weight', 'head.bias', 'end.2.weight', 'end.2.bias']:
                if v in state_dict:
                    del state_dict[v]
        strict=not(transfer)
        model.load_state_dict(state_dict, strict=strict)


class LoggerInstance():
    def __init__(self, typ, epoch, tb_writer):
        self.typ = typ
        self.epoch = epoch
        self.sum_loss = 0.0
        self.N = 0
        self.N_loss = 0
        self.sum_err_3d = 0.0
        self.sum_err_depth = 0.0
        self.sum_err_3d_d_norm = 0.0
        self.sum_err_ang = 0.0
        self.sum_err_ortho = 0.0
        self.sum_err_rel_depth = 0.0
        self.apck = 0.0
        self.average_az = 0.0
        self.average_A = 0.0
        self.tb_writer = tb_writer

    @staticmethod
    def angle(x, y):
        return np.abs(np.arccos(np.clip(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)), -1 ,1)))

    def add(self, losses, mu_v, mu_z, v, annotated, valid, az, A):
        losses = losses[valid]
        mu_v = mu_v[valid]
        mu_z = mu_z[valid]
        v = v[valid]
        annotated = annotated[valid]
        B = mu_v.shape[0]
        N = mu_v.shape[1]
        self.sum_loss += np.sum(losses)
        self.N_loss += len(losses)
        for batch_idx in range(B):
            for kp_idx in range(N):
                if not annotated[batch_idx, kp_idx]:
                    continue
                v_pred = np.empty((3))
                v_pred[:2] = mu_v[batch_idx,kp_idx]*mu_z[batch_idx,kp_idx]
                v_pred[2] = mu_z[batch_idx,kp_idx]
                v_cur = v[batch_idx, kp_idx]
                err_3d = v_cur-v_pred
                err_norm = np.linalg.norm(err_3d)
                self.apck += err_norm < 150
                self.sum_err_3d += err_norm
                vn = np.linalg.norm(v_cur)
                self.sum_err_3d_d_norm += err_norm/vn
                vpn = np.linalg.norm(v_pred)
                v_pred_proj = np.sum(v_cur*v_pred)*v_cur/(vn**2)
                depth_diff = np.abs(np.linalg.norm(v_pred_proj)-vn)
                self.sum_err_depth += depth_diff
                err_orthogonal = np.linalg.norm(v_pred_proj - v_pred)
                self.sum_err_ortho += err_orthogonal
                angle = self.angle(v_pred, v_cur)
                self.sum_err_ang += angle
                self.sum_err_rel_depth += depth_diff / vn
                self.average_az += az[batch_idx, kp_idx]
                self.average_A += np.linalg.det(A[batch_idx, kp_idx])
                self.N += 1

    def finish(self):
        print('{} epoch: {}'.format(self.typ, self.epoch))
        loss = self.sum_loss/self.N_loss
        print('loss: {}'.format(loss))
        err_3d = self.sum_err_3d/self.N
        print('err_3d: {}'.format(err_3d))
        err_3d_normalized = 100*self.sum_err_3d_d_norm/self.N
        print('err_3d / ||v||: {}%'.format(err_3d_normalized))
        angle = self.sum_err_ang/self.N
        print('err_ang: {}'.format(angle))
        err_depth = self.sum_err_depth/self.N
        print('err_depth: {}'.format(err_depth))
        err_rel_depth = 100*self.sum_err_rel_depth/self.N
        print('err_depth / ||v||: {}%'.format(err_rel_depth))
        err_ortho = self.sum_err_ortho/self.N
        print('err ortho: {}'.format(err_ortho))
        apck = self.apck/self.N
        print('err_apck: {}'.format(apck))
        average_az = self.average_az/self.N
        print('average a: {}'.format(average_az))
        average_A = self.average_A/self.N
        print('average |A|: {}'.format(average_A))
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('{}/loss'.format(self.typ), loss, self.epoch)
            self.tb_writer.add_scalar('{}/err_3d'.format(self.typ), err_3d, self.epoch)
            self.tb_writer.add_scalar('{}/relative_err_3d'.format(self.typ), err_3d_normalized, self.epoch)
            self.tb_writer.add_scalar('{}/angle'.format(self.typ), angle, self.epoch)
            self.tb_writer.add_scalar('{}/err_depth'.format(self.typ), err_depth, self.epoch)
            self.tb_writer.add_scalar('{}/err_rel_depth'.format(self.typ), err_rel_depth, self.epoch)
            self.tb_writer.add_scalar('{}/err_ortho'.format(self.typ), err_ortho, self.epoch)
            self.tb_writer.add_scalar('{}/apck'.format(self.typ), apck, self.epoch)
            self.tb_writer.add_scalar('{}/average_az'.format(self.typ), average_az, self.epoch)
            self.tb_writer.add_scalar('{}/average_A'.format(self.typ), average_A, self.epoch)
