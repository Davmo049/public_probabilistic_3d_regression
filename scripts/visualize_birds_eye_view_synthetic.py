import os
import matplotlib.pyplot as plt

import Datasets.SytheticDataset.Dataset as Dataset
import numpy as np

from general_utils.environment_variables import get_dataset_dir

def deserialize_vecs(s):
    s = s[1:-1]
    vstrings = s.split(',')
    vs = []
    for vstring in vstrings:
        vstring = vstring[1:-1]
        v = np.array(list(map(float, vstring.split(' '))))
        vs.append(v)
    return np.array(vs)


def deserialize_mats(s):
    v = deserialize_vecs(s)
    return np.array(list(map(lambda x: x.reshape(2,2), v)))

def deser_floats(s):
    s = s[1:-1]
    return np.array(list(map(float, s.split(', '))))

def load_data(s):
    with open(os.path.join('eval_set_dumps', s+'.txt'), 'r') as f:
        lines = f.readlines()
    deser_fun = [deserialize_mats, deser_floats, deserialize_vecs, deser_floats, deserialize_vecs, deser_floats]
    ret = {}
    for l,f in zip(lines,deser_fun):
        l = l[:-1]
        name, data = l.split(':')
        data = f(data)
        ret[name] = data
    return ret

def huber(v):
    m = v > 1
    ret = v**2/2
    ret[m] = v[m] -1/2
    return ret

def compute_log_prob_proj(v,mu,A,a):
    # v is of size 2xN
    # mu is of size 2
    # A is float
    # a is a float
    vp = v[0]/v[1]
    z = v[1]
    mu_z = mu[1]
    log_p_depth = -a*np.maximum(z/mu_z, mu_z/z)
    mu_v = mu[0]
    norm = np.abs(A*(vp-mu_v))
    log_p_proj = -huber(norm*z/mu_z)
    return np.exp(log_p_proj + log_p_depth)

def compute_range(Ap, a, mu_p, mu_z, num_proj_stds=2.0, depth_range_factor=0.90):
    # Ap is float, correspond to A projected on a single axis
    # a is float
    # mu_p is float correspond to mu_v projected on same axis as A
    # mu_z is float
    expihalf = np.exp(-0.5)
    xdz_min = mu_p - num_proj_stds*np.sqrt((4+3*expihalf)/(2+2*expihalf))/Ap
    xdz_max = mu_p + num_proj_stds*np.sqrt((4+3*expihalf)/(2+2*expihalf))/Ap
    depth_range_factor *= a
    z_max = depth_range_factor*mu_z/a
    z_min = mu_z/depth_range_factor*a
    x_min = min(z_min*xdz_min, z_max*xdz_min)
    x_max = max(z_min*xdz_max, z_max*xdz_max)
    return (x_min, x_max), (z_min, z_max)


def extract_data(ds, saved_output, idx):
    A = saved_output['A'][idx]
    az = saved_output['az'][idx]
    mu_v = saved_output['mu_v'][idx]
    mu_z = saved_output['mu_z'][idx]
    mu = np.empty((3))
    mu[:2] = mu_v
    mu[2] = mu_z
    v = saved_output['v'][idx]
    f = saved_output['f'][idx]
    im, f_ref, v_ref,_,_ = ds[idx]
    im = im[::-1]
    im = im.transpose(1,2,0)
    im = np.clip(im, 0, 1)
    return im, A,az, mu,f,v

def draw_ellipse(ax, center, A, r):
    Ai = np.linalg.inv(A)
    N = 100
    angles = np.arange(N+1)/N*np.pi*2
    xy = np.stack((np.cos(angles), np.sin(angles)), axis=0)
    ell_shape = np.matmul(Ai, xy)*r
    circle = ell_shape + center.reshape(2,1)
    ax.plot(circle[0], circle[1], 'g')

def plot_proj_level_curves(ax, im, A,mu, f,v):
    # radii = [1/np.sqrt(2), 1, 5/4, 6/4, 7/4,2]
    radii = np.arange(1, 3)*2+1/2
    ax.imshow(im)
    ax.plot([mu[0]*f+112], [mu[1]*f+112], 'gx')
    ax.plot([v[0]/v[2]*f+112], [v[1]/v[2]*f+112], 'ro')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for r in radii:
        draw_ellipse(ax, mu[:2]*f+112, A/f, r)

def plot_depth_level_curves(ax, A,az,mu,v):
    x = v[0]
    z = v[1]
    Ap = A[0,0]
    mu_x = mu[0]
    mu_z = mu[2]
    xrange, zrange = compute_range(Ap, az, mu_x, mu_z)
    Nx = 201
    Nz = 201
    xvals = np.arange(Nx)/(Nx-1)*(xrange[1]-xrange[0])+xrange[0]
    zvals = np.arange(Nz)/(Nz-1)*(zrange[1]-zrange[0])+zrange[0]
    xv, zv = np.meshgrid(xvals, zvals)
    grid = np.stack((xv,zv), axis=0)
    flatgrid = grid.reshape(2,-1)
    mup = np.stack((mu_x, mu_z))
    logp = compute_log_prob_proj(flatgrid,mup,Ap,az).reshape(xv.shape)
    ax.contour(xvals, zvals, logp)
    ax.plot([v[0]], [v[2]], 'ro')
    ax.plot([mu_x*mu_z], [mu_z], 'gx')
    ax.set_aspect(0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.get_xaxis().set_ticks([])

def main():
    # subset = 'no_occlusion'
    subset = 'default'
    # subset = 'scale_variation'
    mapping_from_subset_to_saved_data = {'default':'occlusion',
                                         'no_occlusion':'no_occlusion',
                                         'scale_variation': 'scale_variation_occlusion',
                                         'scale_variation_no_occlusion': 'scale_variation_no_occlusion'}
    path = os.path.join(get_dataset_dir(), Dataset.SYNTHETIC_DATA_PATH, subset)
    saved_output = load_data(mapping_from_subset_to_saved_data[subset])

    dataset = Dataset.Dataset(path, normalize_image=False)
    eval_ds = dataset.get_eval()
    for idx in range(200):
        im, A,az,mu,f,v = extract_data(eval_ds, saved_output, idx)
        fig, axs = plt.subplots(1,3)
        plot_proj_level_curves(axs[0], im, A, mu,f,v)
        plot_depth_level_curves(axs[1], A, az, mu, v)
        plot_depth_level_curves(axs[2], A, az, mu, v)
        axs[2].plot([0],[0],'bo')
        fig.subplots_adjust(wspace=0.01)
        plt.show()

if __name__ == '__main__':
    main()
