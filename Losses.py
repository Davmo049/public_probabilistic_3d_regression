import numpy as np

import torch
import mathlib.torch_math as torch_math


NU_V_B_SIZE = 5
START_INDEX_NU_V_B = 0
END_INDEX_NU_V_B = START_INDEX_NU_V_B+NU_V_B_SIZE
DIST_PARAM_SIZE = 2
START_DIST_PARAM = END_INDEX_NU_V_B
END_DIST_PARAM = START_DIST_PARAM + DIST_PARAM_SIZE

class DepthLoss():
    def __init__(self, muz0, device, dtype):
        # D has shape K
        # muz0 has shape K
        self.muz0 = torch.tensor(muz0, device=device, dtype=dtype).view(1,-1)

    def __call__(self, w, z, f, annotated_keypoints, valid_depth):
        # z is NxK
        # f is N
        # ann kp is NxK
        # valid_depth is N
        assert(len(z.shape) == 2)
        N = z.shape[0]
        K = z.shape[1]
        assert(len(w.shape) == 3)
        assert(w.shape[0] == N)
        assert(w.shape[1] == K)
        assert(w.shape[2] == 2)
        assert(len(f.shape) == 1)
        assert(f.shape[0] == N)
        assert(len(annotated_keypoints.shape) == 2)
        assert(annotated_keypoints.shape[0] == N)
        assert(annotated_keypoints.shape[1] == K)
        assert(len(valid_depth.shape) == 1)
        assert(valid_depth.shape[0] == N)

        z_p = z/(f.view(-1,1)*self.muz0)
        z_p[torch.logical_not(annotated_keypoints)] = 1
        z_p[torch.logical_not(valid_depth), :] = 1

        u = w[:,:,0]
        v = w[:,:,1]

        # create a, nu
        mask_v = v < 0
        a = v+1
        a[mask_v] = torch.exp(v[mask_v])
        mm=u<0
        mp=u>=0
        nu = torch.empty(u.shape, dtype=u.dtype, device=u.device)
        nu[mp] = u[mp]/a[mp]+1
        nu[mm] = 1/(1-u[mm]/a[mm])

        # create loss
        depth_loss_err = a*torch.maximum(nu/z_p, z_p/nu)
        depth_loss_normalizing = torch_math.logGammaSymm1(a) + torch.log(nu)
        depth_loss = depth_loss_err + depth_loss_normalizing
        depth_loss[torch.logical_not(annotated_keypoints)] = 0
        depth_loss[torch.logical_not(valid_depth), :] = 0
        a, mu = self.to_original_basis(a, nu, f)
        return a, mu, depth_loss

    def to_original_basis(self, a, nu, f):
        mu = nu*f.view(-1,1)*self.muz0
        return a, mu


class Loss3d():
    def __init__(self, mu_z0, S, device='cpu', dtype=torch.float32):
        self.mu_z0 = torch.tensor(mu_z0, device=device, dtype=dtype).view(1,-1)
        self.depth_loss = DepthLoss(mu_z0, device, dtype)
        self.proj_loss = ProjLoss(mu_z0, S, device, dtype)
        self.S = S

    def __call__(self, targets, f, o, annotated_keypoints, valid_depth):
        """
          v is BxNxEND_DIST_PARAM
          targets is BxNx3
             last dim of targets is x_p, y_p, z_p
        """
        N = o.shape[0]
        K = o.shape[1]
        w_proj = o[:, :, START_INDEX_NU_V_B:END_INDEX_NU_V_B]

        w_dist = o[:, :, START_DIST_PARAM:END_DIST_PARAM]

        a_z, mu_z, depth_loss = self.depth_loss(w_dist, targets[:,:,2], f, annotated_keypoints, valid_depth)

        proj_loss, mu_v, B = self.proj_loss(w_proj, targets, f, annotated_keypoints, valid_depth)
        A = B*(2/self.S*mu_z.view(N,K)/self.mu_z0).view(N,K,1,1)

        losses = depth_loss + proj_loss
        losses = torch.sum(losses, dim=1)
        losses = losses /  o.shape[1]
        return losses, A, a_z, mu_v, mu_z.view(N,K,1)

    def only_remap(self, f, o):
        v_3p = torch.zeros((o.shape[0], o.shape[1], 3), device=o.device, dtype=o.dtype)
        annotated = torch.zeros((o.shape[0], o.shape[1]), device=o.device, dtype=o.dtype)
        valid_depth = torch.zeros((o.shape[0]), device=o.device, dtype=o.dtype)
        losses, A, a_z, mu_v, mu_z = self(v_3p, f, o, annotated, valid_depth)
        return A, a_z, mu_v, mu_z


def abs_term(a, nu, z):
    # shape of all is same
    diff = (a*z-nu)
    abs_diff = torch.abs(diff)
    return abs_diff


def huber_term(A, x, nu, weights_inside):
        A_shape_in = A.shape
        D = A_shape_in[-1]
        A = A.view(-1, D, D)
        N = A.shape[0]
        x = x.view(N, D, 1)
        nu = nu.view(N, D, 1)
        weights_inside = weights_inside.view(-1)
        diff = (torch.bmm(A,x)-nu).view(N,D)
        term = weights_inside*torch.matmul(diff.view(N,1,D), diff.view(N,D,1)).view(N)/2
        mask = term > 1/2
        term[mask] = torch.norm(diff[mask], dim=1)-0.5
        return term.view(A_shape_in[:-2])


class ProjLoss():
    def __init__(self, mu_z0, S, device, dtype):
        # mu_z0 is 1xK
        super().__init__()
        theta = 1.0
        self.S = S
        self.mu_z0 = torch.tensor(mu_z0, dtype=dtype, device=device).view(1,-1,1)
        self.remapping = torch_math.create_posdef_symeig_remap(theta)

    def __call__(self, w, targets, f, annotated_points, valid_depth):
        # values is NxKx5
        # targets is NxKx3
        # f is N
        # annotated_points is NxK

        # return loss, mean (BxNx2), precision (BxNx2x2)
        # do not backprop w.r.t modes, possible numerical instability due to
        # torch.solve
        N = w.shape[0]
        K = w.shape[1]
        z = targets[:,:,2].view(N,K,1)
        f = f.view(-1,1,1)
        v_p = targets[:,:,:2]*((2/self.S)*f/z)
        v_p[torch.logical_not(annotated_points),:] = 0
        weights_inside = (z.view(N,K,1)/(f*self.mu_z0))
        weights_inside[torch.logical_not(valid_depth),:] = 1
        nu = w[:, :, :2]
        B_pre_activation = torch.empty((N, K, 2, 2), dtype=w.dtype, device=w.device)
        B_pre_activation_params = w[:, :, 2:]
        B_pre_activation[:, :, 0,0] = B_pre_activation_params[:, :, 0]
        B_pre_activation[:, :, 0,1] = B_pre_activation_params[:, :, 1] * 0.707106
        B_pre_activation[:, :, 1,0] = B_pre_activation_params[:, :, 1] * 0.707106 # unnecessary symeig only cares about upper region (parameter upper=True)
        B_pre_activation[:, :, 1,1] = B_pre_activation_params[:, :, 2]
        eigs, B = self.remapping(B_pre_activation)

        reg_loss_per_kp = huber_term(B, v_p, nu, weights_inside)
        det_loss_per_kp = -torch.sum(torch.log(eigs), dim=2)
        loss = reg_loss_per_kp + det_loss_per_kp
        loss[torch.logical_not(annotated_points)] = 0
        nu_solve = nu.view(N*K,2)
        B_solve = B.view(N*K,2,2)
        try:
            modes = torch.linalg.solve(B_solve, nu_solve)
            modes = modes.view(N,K,2)
        except RuntimeError as e:
            print("got error in solve")
            print(e)
            # got CUDA error: misaligned address before
            # write stuff for debug
            # print(e)
            # print(nu.dtype)
            # print(nu.device)
            # print(half_prec.dtype)
            # print(half_prec.device)
            modes = torch.zeros((N,K,2), dtype=values.dtype, device=values.device)
        mu = modes*self.S/(2*f)
        return loss, mu, B

class BaselineLoss():
    def __init__(self, mean, std, device, S, squared=False, euclidean=False, logz=False):
        self.mean = mean.view(1,1,3).to(device)
        self.std = std.view(1,1,3).to(device)
        self.S = S
        self.squared = squared
        self.euclidean = euclidean
        self.logz = logz

    def __call__(self, gt, f, o, annotated, valid_depth):
        # gt is NxKx3
        # f is N
        # o is NxKx3
        N = gt.shape[0]
        K = gt.shape[1]
        f = f.view(N,1,1)
        A = torch.zeros((N,K,2,2),device=gt.device, dtype=gt.dtype)
        A[:,:,0,0] = 1
        A[:,:,1,1] = 1
        a = torch.ones((N,K), device=gt.device, dtype=gt.dtype)
        if self.euclidean:
            normalized = (gt-self.mean)/self.std
            if self.squared:
                losses = torch.sum((normalized-o)**2,dim=2)
            else:
                losses = torch.norm(normalized-o,dim=2)
            pred_o = o*self.std+self.mean
            mu_v = pred_o[:,:,:2]/pred_o[:,:,2].view(N,K,1)
            mu_z = pred_o[:,:,2]
        else:
            gt_p = torch.empty(gt.shape, dtype=gt.dtype, device=gt.device)
            gt_p[:,:,:2] = 2*f*gt[:,:,:2]/(self.S*gt[:,:,2].view(N,K,1))
            if self.logz:
                gt_p[:,:,2] = torch.log(gt[:,:,2]/(f[:,:,0]*self.mean[:,:,2]))/self.std[:,:,2]
            else:
                gt_p[:,:,2] = gt[:,:,2]/(f[:,:,0]*self.mean[:,:,2])-1
            if self.squared:
                losses = torch.sum((gt_p-o)**2,dim=2)
            else:
                losses = torch.norm(gt_p-o,dim=2)
            mu_v = self.S*o[:,:,:2]/(f*2)
            if self.logz:
                mu_z = torch.exp(o[:,:,2]*self.std[:,:,2])*f[:,:,0]*self.mean[:,:,2]
            else:
                mu_z = (o[:,:,2]+1)*f[:,:,0]*self.mean[:,:,2]
        losses = torch.mean(losses, dim=1)
        return losses, A, a, mu_v, mu_z
