import numpy as np
from matplotlib import pyplot as plt
from common import homography_transform


def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    XY = np.insert(XY, 2, 1, axis=1)
    #XY = np.insert(XY, 5, 1, axis=1)
    xy = XY[:, 0:3]
    xyprime = XY[:, 3:5]
    zeros = np.zeros((1,4))
    A = np.zeros((2 * XY.shape[0], 9))
    b = np.zeros((2 * XY.shape[0], 1))
    ind = 0
    for x in np.arange(0, 2 * XY.shape[0], 2):
        A[x, 0:3] = zeros
        A[x, 3:6] = -1*xy[ind, :]
        A[x, 6:9] = xyprime[ind, 1]*xy[ind, 1]
        A[x + 1, 0:3] = xy[ind, :]
        A[x + 1, 3:6] = zeros
        A[x + 1, 6:9] = xyprime[ind, 0] * xy[ind, :] * -1
        #print(A[x, :])
        #print(A[x+1, :])
        ind = ind + 1
    w, v = np.linalg.eig(np.dot(A.T, A))
    h = v[:,np.argmin(w)]
    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]])/h[8]
    return H


def RANSAC_fit_homography(XY, eps=2, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    subset = np.empty((4,))
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    #E = np.zeros((XY.shape[0],))
    h = np.eye(3)
    bestRefit = np.eye(3)
    for trial in range(nIters):
        subset = XY[np.random.choice(XY.shape[0], 4, replace=False), :]
        h = fit_homography(subset)
        E = np.sum((XY[:, 2:4] - homography_transform(XY[:,0:2], h))**2, axis=1)**0.5
        Inliers = XY[np.transpose(np.where(E<eps))[:,0], :]
        if np.transpose(np.where(E<eps))[:,0].shape[0] > bestCount:
            bestH, bestCount, bestInliers = h, np.transpose(np.where(E<eps))[:,0].shape[0], Inliers
    bestRefit = fit_homography(bestInliers)
    return bestRefit


if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    data = np.load("task4/points_case_9.npy")
    hom = RANSAC_fit_homography(data)
    print(hom)

    hom1 = fit_homography(data)
    print(hom1)

    xy = data[:, 0:2]
    xyprime = data[:, 2:4]

    # 4. plot the original points and transformed points
    plt.scatter(xy[:, 0], xy[:, 1], 1, c='r')
    plt.scatter(xyprime[:, 0], xyprime[:, 1], 1, c='g')
    trans = homography_transform(xy, hom)
    plt.scatter(trans[:, 0], trans[:, 1], 1, c='b')
    plt.show()

    pass
