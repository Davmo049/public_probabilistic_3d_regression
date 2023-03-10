import ctypes
import os
import numpy as np

path_this = os.path.abspath(__file__)
dir_this = '/'+'/'.join(path_this.split('/')[:-1])
so_path = os.path.join(dir_this, 'build/library.so')
loaded_library = ctypes.CDLL(so_path)


c_fill = loaded_library.fill

import matplotlib.pyplot as plt

def cross(v1, v2):
    return np.array((
        v1[1]*v2[2]-v1[2]*v2[1],
        v1[2]*v2[0]-v1[0]*v2[2],
        v1[0]*v2[1]-v1[1]*v2[0]))

def render(image, focal_length, corners, colors, lightvec=None, ambient=None):
    if lightvec is None:
        lightvec = np.array((0,0,1.0),dtype=np.float32)
        # note lightvec is direction
        # note directional light is white
    if ambient is None:
        ambient = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        # note ambient is colored light
    assert(corners.shape[0] == 8)
    assert(corners.shape[1] == 3)
    assert(corners.dtype==np.float32)
    assert(colors.shape[0] == 6)
    assert(colors.shape[1] == 3)
    assert(ambient.dtype == np.float32)
    assert(ambient.shape[0] == 3)

    pp = np.array(image.shape[:2])/2
    faces = [[0,2,3,1], [4,5,7,6], [0,4,6,2], [1,3,7,5], [0,1,5,4], [2,6,7,3]]
    for f, c in zip(faces, colors):
        c0 = corners[f[0]]
        c1 = corners[f[1]]
        c2 = corners[f[3]]
        v = np.mean(corners[f], axis=0)-np.array((0,0,10))
        normal = cross(c1-c0, c2-c0)
        normal /= np.linalg.norm(normal)
        if np.sum(v*normal) < 0:
            print(c1-c0)
            print(c2-c0)
            print(corners[f])
            print(v)
            print(normal)
            print(f)
        if np.sum(c0*normal[2]) < 0:
            facecolor = (max(0, -np.sum(normal*lightvec))+ambient)*c
            facecorners = corners[f]
            facecorners = facecorners[:, :2]/facecorners[:, 2].reshape(-1,1)
            corners_2d = (facecorners * focal_length + pp.reshape(-1, 2)).astype(np.int32)
            fill(image, corners_2d, facecolor)

def fill(image, facecorners, facecolor):
    assert(image.dtype==np.float32)
    assert(image.shape[2] == 3)
    assert(facecorners.shape[0] == 4)
    assert(facecorners.shape[1] == 2)
    assert(facecorners.dtype == np.int32)
    assert(facecolor.shape[0]==3)
    assert(facecolor.dtype==np.float32)
    R = image.shape[1]
    C = image.shape[0]
    pixbuf = (ctypes.c_float*(R*C*3)).from_buffer(image.data)

    cornerbuf = (ctypes.c_float*(8)).from_buffer(facecorners.data)
    colorbuf = (ctypes.c_float*(3)).from_buffer(facecolor.data)
    c_fill(pixbuf, R, C, cornerbuf, colorbuf)
    ###pixbuf modified, no return

def main_fill():
    im = np.zeros((200, 200, 3), dtype=np.float32)
    c1 = [100, 0]
    c2 = [0, 100]
    c3 = [100, 200]
    c4 = [200, 101]
    corners = [c4,c1,c2,c3]
    corners = np.array(corners, dtype=np.int32)
    color = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    import time
    s = time.time()
    T = 30000
    for _ in range(T):
        fill(im, corners, color)
    print((time.time()-s)/T)
    plt.imshow(im)
    plt.show()

def main_box():
    def mat_from_quat(q):
        R = np.empty((3,3), dtype=np.float32)
        R[0,0] = 1-2*(q[2]**2+q[3]**2)
        R[1,1] = 1-2*(q[1]**2+q[3]**2)
        R[2,2] = 1-2*(q[1]**2+q[2]**2)
        R[0,1] = 2*(q[1]*q[2]-q[3]*q[0])
        R[0,2] = 2*(q[1]*q[3]+q[2]*q[0])
        R[1,0] = 2*(q[1]*q[2]+q[3]*q[0])
        R[2,0] = 2*(q[1]*q[3]-q[2]*q[0])
        R[1,2] = 2*(q[2]*q[3]-q[1]*q[0])
        R[2,1] = 2*(q[2]*q[3]+q[1]*q[0])
        return R

    image = np.ones((224, 224, 3), dtype=np.float32)*0.5
    q = np.random.normal(size=(4))
    # q = np.array([1.0,0,0,0])
    q /= np.linalg.norm(q)
    R = mat_from_quat(q)
    basic_corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                basic_corners.append(np.array((k,j,i)))
    # faces
    # 0,2,3,1
    # 4,5,7,6
    # 0,4,6,2
    # 1,3,7,5
    # 0,1,5,4
    # 2,6,7,3

    basic_corners = np.array(basic_corners, dtype=np.float32)
    corners = np.array((0.0,0,10)).reshape(1,3) + np.matmul(R, basic_corners.transpose()).transpose()
    corners = np.ascontiguousarray(corners.astype(np.float32))
    colors = np.array([[1.0, 0,0],[0,1.0, 0], [0,0,1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [0,1.0,1.0]], dtype=np.float32)
    # colors = np.ones((6,3), dtype=np.float32)/2
    ambient = np.array([0.1, 0.1, 0.1])
    import time
    s = time.time()
    T = 10000
    for i in range(T):
        render(image, 500.0, corners, colors, lightvec=np.array((0,0,1.0),dtype=np.float32), ambient=np.array([0.1, 0.1, 0.1], dtype=np.float32))
    print((time.time()-s)/T)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main_box()
    # main_fill()
