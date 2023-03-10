import os
import matplotlib.pyplot as plt

import numpy as np

def IncompleteGammaIntp(k,x):
    # computes exp(x)Gamma(k,x)/x^k
    # useful when k >= 0
    N = 512
    q = (np.arange(N)+1)/N
    w = np.ones((N))
    w[-1] = 1/2
    y = (1-(k+1)/x*np.log(q))**(k-1)*q**k
    return np.sum(y*w)*(k+1)/x/N

def IncompleteGammaIntm(k,x):
    # computes exp(x)Gamma(k,x)/x^k
    # useful when k <= 0
    N = 512
    q = (np.arange(N)+1)/N
    w = np.ones((N))
    w[-1] = 1/2
    y = (1-1/x*np.log(q))**(k-1)
    return np.sum(y*w)/x/N


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

def integral_values(v,N):
    cv = np.cumsum(v)
    return (cv[N:]-cv[:-N])/N


def bin_values(v, N):
    assert(len(v)%N ==0)
    r = []
    for i in range(len(v)//N):
        r.append(np.mean(v[i*N:(i+1)*N]))
    return np.array(r)

def compute_expected_err(a, mu_z):
    lvar = []
    le = []
    for v in a:
        e, var = e_var_single_a(v)
        le.append(e)
        lvar.append(var)
    le = np.array(le)
    lvar = np.array(lvar)
    return mu_z*le, mu_z**2*lvar

def compute_err_z(dic):
    mu_z = dic['mu_z']
    z = dic['v'][:,2]
    a = dic['az']
    muz, varz = compute_expected_err(a,mu_z)
    diff = (z-muz)**2
    index = np.argsort(varz)
    expected_err = varz[index]
    diff = diff[index]
    N = 100
    ee_binned = integral_values(expected_err, N)
    diff_binned = integral_values(diff, N)
    return ee_binned, diff_binned


def e_var_single_a(a):
    f1 = (IncompleteGammaIntp(1, a) + IncompleteGammaIntm(-1, a))
    f2 = (IncompleteGammaIntp(2, a) + IncompleteGammaIntm(-2, a))
    f3 = (IncompleteGammaIntp(3, a) + IncompleteGammaIntm(-3, a))
    return f2/f1, (f3*f1-f2**2)/f1**2

print(e_var_single_a(1.0))

def expected_error_A(As):
    ee = []
    for A in As:
        ev, _ = np.linalg.eig(A)
        ee.append(np.sum(1/ev**2))
    return np.array(ee)*(4+3*np.exp(-1/2))/(2+2*np.exp(-1/2))


def compute_err_p(dic):
    A = dic['A']
    mu_v = dic['mu_v']
    f = dic['f']
    v = dic['v']
    ee = expected_error_A(A)
    vp = v[:,:2]/v[:,2].reshape(-1,1)
    diff = np.sum((vp-mu_v)**2,axis=1)
    index = np.argsort(ee)
    expected_err = ee[index]
    diff = diff[index]
    N = 100
    ee_smooth = integral_values(expected_err, N)
    diff_smooth = integral_values(diff, N)
    return ee_smooth, diff_smooth





def main():
    dataset = ['no_occlusion', 'occlusion', 'scale_variation', 'scale_variation_occlusion']
    color = ['r','g', 'b','k']
    fig, axs = plt.subplots(1,2)
    axs[0].set_xlabel('Expected analytical squared depth error')
    axs[0].set_ylabel('Empirical squared depth error')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xlabel('Expected analytical projected squared projected error')
    axs[1].set_ylabel('Empirical squared projected error')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    min_vd = np.inf
    max_vd = -np.inf
    min_vp = np.inf
    max_vp = -np.inf
    for d, c in zip(dataset, color):
        dic = load_data(d)
        ee_smooth, diff_smooth = compute_err_z(dic)
        if min(ee_smooth[0], np.min(diff_smooth)) < min_vd:
            min_vd = min(ee_smooth[0], np.min(diff_smooth))
        if max(ee_smooth[-1], np.max(diff_smooth)) > max_vd:
            max_vd = max(ee_smooth[-1], np.max(diff_smooth))
        axs[0].plot(ee_smooth, diff_smooth,c, label=d)

        ee_smooth, diff_smooth = compute_err_p(dic)
        if min(ee_smooth[0], np.min(diff_smooth)) < min_vp:
            min_vp = min(ee_smooth[0], np.min(diff_smooth))
        if max(ee_smooth[-1], np.max(diff_smooth)) > max_vp:
            max_vp = max(ee_smooth[-1], np.max(diff_smooth))
        axs[1].plot(ee_smooth, diff_smooth,c)



    axs[0].plot([min_vd, max_vd], [min_vd, max_vd], 'r--')
    axs[1].plot([min_vp, max_vp], [min_vp, max_vp], 'r--')
    axs[0].legend()
    plt.show()


if __name__ == '__main__':
    main()
