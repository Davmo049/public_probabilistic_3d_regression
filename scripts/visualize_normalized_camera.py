import numpy as np
import matplotlib.pyplot as plt

def main():
    box_width = 30
    focal_length = 1000
    principal_point = np.array((300.0, 300))
    distance = 100
    basic_box = box_width/2 * np.array(((1,1,1.0), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1),(-1,1,-1),(-1,-1,1), (-1,-1,-1)))
    box1 = np.array(((0,0,distance))) + basic_box
    c30 = np.cos(30/180*np.pi)
    s30 = np.sin(30/180*np.pi)
    R = np.array(((c30, 0, s30), (0,1,0), (-s30, 0, c30)))
    # box2 = np.matmul(R,box1.transpose()).transpose()
    # bbx1 = draw_box(box1, focal_length, principal_point)
    # bbx2 = draw_box(box2, focal_length, principal_point, 'g')
    # plt.plot((principal_point[0]), (principal_point[1]), 'rx')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim(0, 1300)
    # plt.ylim(0, 600)
    # plt.show()
    # draw_birds_eye(box1, 'b')
    # draw_birds_eye(box2, 'g')
    # draw_birds_camera()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim(np.min(box1[:, 0])-5, np.max(box2[:, 0])+5)
    # plt.ylim(-5, np.max(box1[:, 2])+5)
    # plt.show()

    # # crop box1
    # draw_box(box1, focal_length, principal_point, draw_bbx=False)
    # plt.xlim(bbx1[0], bbx1[1])
    # plt.ylim(bbx1[2], bbx1[3])
    # plt.show()
    # plt.gca().set_aspect('equal', adjustable='box')
    # draw_birds_eye(box1, 'b')
    # draw_birds_camera()
    # plt.show()

    # draw_birds_eye(box1, 'g')
    # plt.gca().set_aspect('equal', adjustable='box')
    # draw_birds_camera()
    # plt.show()

    # # crop box 2
    # draw_box(box2, focal_length, principal_point, 'g', draw_bbx=False)
    # plt.xlim(bbx2[0], bbx2[1])
    # plt.ylim(bbx2[2], bbx2[3])
    # plt.show()
    # # second case
    # rot_box = np.matmul(np.linalg.inv(R),basic_box.transpose()).transpose()
    # box3 = np.array(((0,0,distance))) + rot_box
    # box4 = np.matmul(R,box3.transpose()).transpose()
    # bbx3 = draw_box(box3, focal_length, principal_point)
    # bbx4 = draw_box(box4, focal_length, principal_point, 'g')
    # plt.plot((principal_point[0]), (principal_point[1]), 'rx')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim(0, 1300)
    # plt.ylim(0, 600)
    # plt.show()
    # draw_birds_eye(box3, 'b')
    # draw_birds_eye(box4, 'g')
    # draw_birds_camera()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim(np.min(box3[:, 0])-5, np.max(box4[:, 0])+5)
    # plt.ylim(-5, np.max(box3[:, 2])+5)
    # plt.show()
    # #crop box 3
    # draw_box(box3, focal_length, principal_point, 'g', draw_bbx=False)
    # plt.xlim(bbx3[0], bbx3[1])
    # plt.ylim(bbx3[2], bbx3[3])
    # plt.show()
    # draw_birds_eye(box3, 'b')
    # draw_birds_camera()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    # # crop box 4
    # draw_box(box4, focal_length, principal_point, 'g', draw_bbx=False)
    # plt.xlim(bbx4[0], bbx4[1])
    # plt.ylim(bbx4[2], bbx4[3])
    # plt.show()
    # draw_birds_eye(box3, 'g')
    # draw_birds_camera()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

    # case 3
    skew_box = np.array(((1,1,1.0), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1),(-1,1,-1),(-1,-1,1), (-1,-1,-1)))
    skew_box[:,2] -= 0.6+skew_box[:,0]*0.4
    skew_box *= box_width/2

    box5 = np.array(((0,0,distance))) + skew_box
    box6 = np.matmul(R,box5.transpose()).transpose()
    bbx5 = draw_box(box5, focal_length, principal_point)
    bbx6 = draw_box(box6, focal_length, principal_point, 'g')
    plt.plot((principal_point[0]), (principal_point[1]), 'rx')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 1300)
    plt.ylim(0, 600)
    plt.show()
    draw_birds_eye(box5, 'b')
    draw_birds_eye(box6, 'g')
    draw_birds_camera()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(np.min(box5[:, 0])-5, np.max(box6[:, 0])+5)
    plt.ylim(-5, np.max(box5[:, 2])+5)
    plt.show()
    #crop box 5
    draw_box(box5, focal_length, principal_point, 'b', draw_bbx=False)
    plt.xlim(bbx5[0], bbx5[1])
    plt.ylim(bbx5[2], bbx5[3])
    plt.show()
    draw_birds_eye(box5, 'b')
    draw_birds_camera()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    # crop box 6
    draw_box(box6, focal_length, principal_point, 'g', draw_bbx=False)
    plt.xlim(bbx6[0], bbx6[1])
    plt.ylim(bbx6[2], bbx6[3])
    plt.show()
    draw_birds_eye(box5, 'g')
    draw_birds_camera()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
 

def draw_box(box, f, pp,c='b', draw_bbx=True):
    edges = ((0,1), (0,2), (0,4), (1,3), (1, 5), (2, 3), (2,6), (3, 7), (4,5), (4,6), (5, 7), (6,7))
    box_hom = box[:, :2]/box[:,2].reshape(-1,1)*f+pp
    for i, j in edges:
        plt.plot((box_hom[i,0], box_hom[j,0]), (box_hom[i,1], box_hom[j,1]), c)
    xmin = np.min(box_hom[:,0])-5
    xmax = np.max(box_hom[:,0])+5
    ymin = np.min(box_hom[:,1])-5
    ymax = np.max(box_hom[:,1])+5
    plt.plot((xmin, xmax, xmax, xmin, xmin), (ymin, ymin, ymax, ymax, ymin),'r')
    return (xmin, xmax, ymin, ymax)

def draw_birds_eye(box, c='r'):
    edges = ((0,1), (0,2), (0,4), (1,3), (1, 5), (2, 3), (2,6), (3, 7), (4,5), (4,6), (5, 7), (6,7))
    for i, j in edges:
        plt.plot((box[i,0], box[j,0]), (box[i,2], box[j,2]), c)

def draw_birds_camera():
    length = 10
    angle = 45
    ca = np.cos(angle*np.pi/180)
    sa = np.sin(angle*np.pi/180)
    plt.plot((-sa*length, 0, sa*length), (ca*length, 0, ca*length),'r')


if __name__ == '__main__':
    main()
