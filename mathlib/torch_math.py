import torch
import numpy as np


def numdiff(f, X, eps=10e-5, batch_dims=1):
    # f is function
    # X is input
    # eps is step to use
    # batch dim is number of first dimensions to ignore as part of a batch (independent)
    X = X.detach()
    bs = np.prod(X.shape[:batch_dims])
    dims_per_batch = np.prod(X.shape[batch_dims:])
    X_like = X.view(bs, dims_per_batch)
    fx = f(X).view(bs)
    diff = torch.empty((bs, dims_per_batch), dtype=X.dtype, device=X.device)
    for i in range(dims_per_batch):
        X_eps = X_like.clone()
        X_eps[:, i] += eps
        diff[:, i] = (f(X_eps.view(*X.shape)).view(bs)-fx)/eps
    return diff.view(*X.shape)


def symeig_2x2(A):
    # default is slow https://github.com/pytorch/pytorch/issues/22573
    # we solve our special case here
    # A = Mx[[a, c], [c, b]]
    # return same as torch.symeig(upper=True, eigenvectors=True)
    M = A.shape[0]
    a = A[:,0,0]
    b = A[:,1,1]
    c = A[:,0,1]
    m = (a+b)/2
    c2 = c**2
    diff_v = (a-b)/2
    diff_sq = diff_v**2
    sq = torch.sqrt(diff_sq+c2)
    eigvals = torch.empty((M, 2), dtype=A.dtype, device=A.device)
    eigvals[:, 0].copy_(m+sq)
    eigvals[:, 1].copy_(m-sq)
    eigvecs = torch.empty((M, 2, 2), dtype=A.dtype, device=A.device)
    mask0 = sq < 1e-5 # both values very similar, avoid numerical instability
    a1 = a - eigvals[:, 0]
    b1 = b - eigvals[:, 0]
    a2 = a - eigvals[:, 1]
    b2 = b - eigvals[:, 1]
    mask1 = torch.abs(a1) < torch.abs(b1)
    mask2 = torch.abs(a2) < torch.abs(b2)
    # default case (neither mask0/1/2)
    eigvecs[:,0,0] = -c
    eigvecs[:,1,0] = a1
    eigvecs[:,0,1] = -c
    eigvecs[:,1,1] = a2

    # mask1/2 case, but not mask 0
    eigvecs[mask1,0,0] = b1[mask1]
    eigvecs[mask1,1,0] = -c[mask1]
    eigvecs[mask2,0,1] = b2[mask2]
    eigvecs[mask2,1,1] = -c[mask2]

    # mask0 case
    eigvecs[mask0,0,0] = 1
    eigvecs[mask0,0,1] = 0
    eigvecs[mask0,1,0] = 0
    eigvecs[mask0,1,1] = 1
    norm = torch.norm(eigvecs, dim=1)
    eigvecs = eigvecs / norm.view(M,2,1)
    return eigvals, eigvecs


def symeig_forward(ctx, A_in, f):
    m = A_in.shape[-1]
    A = A_in.view(-1, m, m)
    D, V = symeig_2x2(A)
    # D, V = torch.symeig(A, upper=True, eigenvectors=True)
    f_D = f(D)
    ctx.save_for_backward(D,V, f_D)
    Vxf_D = V*f_D.view(-1, 1,m)
    Vxf_DxVt = torch.bmm(Vxf_D, V.transpose(1,2))
    return f_D.view(A_in.shape[:-1]), Vxf_DxVt.view(A_in.shape)

def symeig_backward(ctx, grad_eigen_in, grad_output_in, f, df):
    m = grad_output_in.shape[-1]
    grad_eigen_in = grad_eigen_in.view(-1, m)
    grad_output = grad_output_in.view(-1, m, m)
    grad_output = (grad_output+grad_output.transpose(1,2))/2
    S, V, f_S = ctx.saved_tensors
    compensated_grad = torch.matmul(torch.matmul(V.transpose(1,2), grad_output), V)
    df_ds =  df(S)

    # note when you have identical eigen values you would hope grad_eigen_in is the same for all identical eigenvalues
    # if not the following line is a bug since the eigenvalues shuffle around when increasing any lambda
    compensated_grad.view(-1, m*m)[:,::m+1] += grad_eigen_in
    eig_diff = S.view(-1, m, 1)-S.view(-1, 1, m)
    close_vals = torch.abs(eig_diff) < 10e-3
    incline = (f_S.view(-1, m, 1) - f_S.view(-1, 1, m)) / eig_diff
    close_val_backup = df_ds.view(-1, m, 1).repeat(1,1,m)
    incline[close_vals] = close_val_backup[close_vals]
    dJ_dA_comp = compensated_grad * incline

    return torch.matmul(torch.matmul(V, dJ_dA_comp), V.transpose(1,2)).view(grad_output_in.shape)


def create_posdef_symeig_remap(theta):
    # this function creates an autograd function class in its
    # local scope, which it then returns.
    # this class references the input to this function
    # allowing a torch function Function to essentially get a configurable state
    def f(s):
        mask = s < 0
        ret = s+theta
        ret[mask] = theta*torch.exp(s[mask]/theta)
        return ret

    def df(s):
        mask = s < 0
        ret = torch.ones_like(s)
        ret[mask] = torch.exp(s[mask]/theta)
        return ret

    class TMPCLASS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A_in):
            return symeig_forward(ctx, A_in, f)

        @staticmethod
        def backward(ctx, grad_eigen_in, grad_output_in):
            return symeig_backward(ctx, grad_eigen_in, grad_output_in, f, df)

    return TMPCLASS.apply

def torch_integral(f, v, N):
    with torch.no_grad():
        # computes ret_i = \int_{0}^{1} f(x,v_i)
        # where N is number of trapezoids + 1 per v_i
        # f(0) = 0 implicit
        rangee = torch.arange(N,dtype=v.dtype, device=v.device)
        x = ((rangee+1)/N).view(1, N)
        weights = torch.empty((1, N), dtype=v.dtype, device=v.device).fill_(1)
        weights[0, -1] = 1/2
        y = f(x, v)
        return torch.sum(y*weights, dim=1)/N

def integrandE1sup(y, a):
    # a is Bx1
    # y is 1xN
    return 2*y/(a+2*torch.log(1/y))

class class_logE1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        N = 512
        intv = torch_integral(integrandE1sup, a.view(-1,1), N).view(a.shape)
        ctx.save_for_backward(a, intv)
        return torch.log(intv)-a

    def backward(ctx, grad):
        a, intv = ctx.saved_tensors
        return -grad/(a*intv)


logE1 = class_logE1.apply

def integrandGammaSymm(y,a):
    return 1/(1+torch.log(1/y)/a)**2

class class_logGammaSymm1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        N = 512
        intv = torch_integral(integrandGammaSymm, a.view(-1,1),N).view(a.shape)
        ctx.save_for_backward(a, intv)
        return torch.log(1+intv)-a-torch.log(a)

    @staticmethod
    def backward(ctx, grad):
        a, intv = ctx.saved_tensors
        mult = (1/a-(2+2/a)/(1+intv))
        return grad*mult
logGammaSymm1 = class_logGammaSymm1.apply
