import numpy as np

def find_reasonably_large_square(im):
    im = im > 0.5
    n1 = np.sum(im.flatten() > 0.5)
    if n1 == 0:
        return None
    ex = np.sum(np.sum(im, axis=0)*np.arange(im.shape[1]))/n1
    ey = np.sum(np.sum(im, axis=1)*np.arange(im.shape[0]))/n1
    ex = int(ex)
    ey = int(ey)
    minx = ex
    maxx = ex
    miny = ey
    maxy = ey
    if im[ey, ex] != 1:
        return None
    directions = [True,True,True,True]
    while any(directions):
        d_to_remove = []
        for d in range(4):
            if not directions[d]:
                continue
            if d < 2:
                if d == 0:
                    ynew = miny-1
                else:
                    ynew = maxy+1
                if ynew < 0 or ynew >= im.shape[0]:
                    d_to_remove.append(d)
                    continue
                no_collision = True
                for c in range(minx, maxx+1):
                    if im[ynew, c] == 0:
                        d_to_remove.append(d)
                        no_collision = False
                        break
                if no_collision:
                    if d == 0:
                        miny = ynew
                    else:
                        maxy = ynew
            else:
                if d == 2:
                    xnew = minx-1
                else:
                    xnew = maxx+1
                if xnew < 0 or xnew >= im.shape[1]:
                    d_to_remove.append(d)
                    continue
                no_collision = True
                for r in range(miny, maxy+1):
                    if im[r, xnew] == 0:
                        d_to_remove.append(d)
                        no_collision = False
                        break
                if no_collision:
                    if d == 2:
                        minx = xnew
                    else:
                        maxx = xnew
        for d in d_to_remove:
            directions[d] = False
    return [minx, maxx, miny, maxy]
