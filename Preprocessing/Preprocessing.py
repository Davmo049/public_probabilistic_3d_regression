import numpy as np


def get_transform_to_ideal_camera(desired_size, bbx, real_intrinsic, desired_up=None):
    points = get_back_proj_bbx(bbx, real_intrinsic)
    if desired_up is None:
        desired_up = np.array([0.0,-1,0])
    real_cam_to_ideal_cam, desired_intrinsic = get_desired_camera(desired_size, points, desired_up)
    cam_im_to_ideal_coords = np.matmul(real_cam_to_ideal_cam[:3,:3], np.linalg.inv(real_intrinsic))
    cam_im_to_ideal_im = np.matmul(desired_intrinsic, cam_im_to_ideal_coords)

    return real_cam_to_ideal_cam, desired_intrinsic, cam_im_to_ideal_im


def get_back_proj_bbx(bbx, intrinsic):
    assert(len(bbx) == 4)
    minx = bbx[0]
    maxx = bbx[1]
    miny = bbx[2]
    maxy = bbx[3]
    points = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
    points = points.transpose()
    points_homo = np.ones((3,4))
    points_homo[:2, :] = points
    intrinsic_inv = np.linalg.inv(intrinsic)
    backproj = np.matmul(intrinsic_inv, points_homo)
    backproj /= np.linalg.norm(backproj, axis=0).reshape(1, -1)
    return backproj


def get_desired_camera(desired_imagesize, backproj, desired_up):
    z, radius_3d = get_minimum_covering_sphere(backproj.transpose())
    y = desired_up - np.dot(desired_up, z)*z
    y /= np.linalg.norm(y)
    x = np.cross(y,z)
    R_cam_to_ideal = np.stack([x, y,z], axis=0)
    bp_reproj = np.matmul(R_cam_to_ideal, backproj)
    bp_reproj /= bp_reproj[2,:].reshape(1, -1)
    f = 1/np.max(np.abs(bp_reproj[:2]).reshape(-1))

    intrinsic = np.array([[desired_imagesize*f/2, 0, desired_imagesize/2],
                          [0, desired_imagesize*f/2, desired_imagesize/2],
                          [0, 0, 1]])
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R_cam_to_ideal
    return extrinsic, intrinsic

# duplicated code from Pascal3D
def get_minimum_covering_sphere(points):
    # points = nx3 array on unit sphere
    # returns point on unit sphere which minimizes the maximum distance to point in points
    # uses modified version of welzl
    points = np.copy(points)
    np.random.shuffle(points)
    def sphere_welzl(points, included_points, num_included_points):
        if len(points) == 0 or num_included_points == 3:
            return sphere_trivial(included_points[:num_included_points])
        else:
            p = points[0]
            rem = points[1:]
            cand_mid, cand_rad = sphere_welzl(rem, included_points, num_included_points)
            if np.linalg.norm(p-cand_mid) < cand_rad:
                return cand_mid, cand_rad
            included_points[num_included_points] = p
            return sphere_welzl(rem, included_points, num_included_points+1)
    buf = np.empty((3,3), dtype=np.float)
    return sphere_welzl(points, buf, 0)

# duplicated code
def sphere_trivial(points):
    if len(points) == 0:
        return np.array([1.0, 0,0]), 0
    elif len(points) == 1:
        return points[0], 0
    elif len(points) == 2:
        mid = (points[0] + points[1])/2
        diff = points-mid.reshape(1, -1)
        r = np.max(np.linalg.norm(diff, axis=1))
        return mid, r
    elif len(points) == 3:
        X = np.stack(points, axis=0)
        C = np.array([1,1,1])
        mid = np.linalg.solve(X, C)
        mid /= np.linalg.norm(mid)
        r = np.max(np.linalg.norm(points-mid.reshape(1, -1), axis=1))
        return mid, r
    raise Exception("2d welzl should not need 4 points")
