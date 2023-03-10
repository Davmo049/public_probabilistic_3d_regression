import ctypes
import os
import numpy as np

path_this = os.path.abspath(__file__)
dir_this = '/'+'/'.join(path_this.split('/')[:-1])
so_path = os.path.join(dir_this, 'build/library.so')
loaded_library = ctypes.CDLL(so_path)

c_interp_bilinear = loaded_library.interp_bilinear
c_interp_nearest = loaded_library.interp_nearest

import matplotlib.pyplot as plt

def interp_c_bilinear(image, sample_r, sample_c):
    assert(image.dtype==np.float32)
    sample_r = sample_r.astype(np.float32)
    sample_c = sample_c.astype(np.float32)
    assert(len(sample_r)==len(sample_c))
    R = image.shape[0]
    C = image.shape[1]
    ch = image.shape[2]
    N = sample_r.shape[0]
    pixbuf = (ctypes.c_float*(R*C*ch)).from_buffer(image.data)
    sample_r = (ctypes.c_float*N).from_buffer(sample_r.data)
    sample_c = (ctypes.c_float*N).from_buffer(sample_c.data)
    retbuf = np.empty((ch*N), dtype=np.float32)
    c_retbuf = (ctypes.c_float*(ch*N)).from_buffer(retbuf.data)
    c_interp_bilinear(pixbuf, R, C, ch, sample_r, sample_c, c_retbuf, N)
    return retbuf

def interp_c_nearest(image, sample_r, sample_c):
    assert(image.dtype==np.uint8)
    sample_r = sample_r.astype(np.float32)
    sample_c = sample_c.astype(np.float32)
    assert(len(sample_r)==len(sample_c))
    R = image.shape[0]
    C = image.shape[1]
    N = sample_r.shape[0]
    pixbuf = (ctypes.c_char*(R*C)).from_buffer(image.data)
    sample_r = (ctypes.c_char*N).from_buffer(sample_r.data)
    sample_c = (ctypes.c_char*N).from_buffer(sample_c.data)
    retbuf = np.empty((N), dtype=np.uint8)
    c_retbuf = (ctypes.c_char*(N)).from_buffer(retbuf.data)
    c_interp_nearest(pixbuf, R, C, sample_r, sample_c, c_retbuf, N)
    return retbuf
