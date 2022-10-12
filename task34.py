import numpy as np
from matplotlib import pyplot as plt
from homography import fit_homography, homography_transform

def p3():
# code for Task 3
    # 1. load points X from task3/
    mat = np.load("task3/points_case_2.npy")
    xy = mat[:, 0:2]
    xyprime = mat[:, 2:4]
    # 2. fit a transformation y=Sx+t
    A = np.zeros((2*mat.shape[0],6))
    b = np.zeros((2*mat.shape[0],1))
    ind = 0
    for x in np.arange(0, 2*mat.shape[0], 2):
        A[x, :] = np.array([xy[ind, 0], xy[ind, 1], 0, 0, 1, 0])
        A[x+1, :] = np.array([0, 0, xy[ind, 0], xy[ind, 1], 0, 1])
        b[x,0] = xyprime[ind, 0]
        b[x+1, 0] = xyprime[ind, 1]
        ind = ind + 1
    # 3. transform the points
    v = np.linalg.lstsq(A, b, rcond=None)[0]
    print(v)
    # 4. plot the original points and transformed points
    plt.scatter(xy[:,0], xy[:,1], 1, c='r')
    plt.scatter(xyprime[:,0], xyprime[:,1], 1, c='g')
    calcx = v[0]*xy[:,0] + v[1]*xy[:,1] + v[4]
    calcy = v[2]*xy[:,0] + v[3]*xy[:,1] + v[5]
    plt.scatter(calcx, calcy, 1, c='b')
    plt.show()
def p4():
    # code for Task 4

    pass

if __name__ == "__main__":
    # Task 3
    p3()

    # Task 4
    p4()