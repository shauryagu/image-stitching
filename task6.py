"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2

def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    N = np.square(np.linalg.norm(desc1, axis=1, keepdims=True)) + np.square(np.linalg.norm(desc2, axis=1, keepdims=True)).T
    dist = N - 2 * (np.dot(desc1, desc2.T))
    dist[dist < 0] = 0
    return dist

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches 
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    D = compute_distance(desc1, desc2)
    ind = np.argsort(D, axis=1)
    S = np.take_along_axis(D, ind, axis=1)
    matches = np.empty((1,2), dtype=int)
    for i in range(S.shape[0]):
        if S[i, 0]/S[i, 1] <= ratioThreshold:
            matches = np.vstack((matches, np.array([i, ind[i, 0]])))
    matches = matches[1:, :]
    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints
    Hint: see cv2.line
    '''
    #Hint:
    #Use common.get_match_points() to extract keypoint locations
    matchLoc = common.get_match_points(kp1, kp2, matches)
    output = np.vstack((img1, img2))
    for r in range(30):
        output = cv2.line(output, (int(matchLoc[r, 0]), int(matchLoc[r, 1])),
                                   (int(matchLoc[r, 2]), int(matchLoc[r, 3] + img1.shape[0])), color=(0,255,0), thickness=2)

    return output


def warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.
    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them
    Output - V: stitched image of size (?,?,3); unknown since it depends on H
    '''
    V = None
    return V


def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
    
    Output - Final stitched image
    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''
    kp1, desc1 = common.get_AKAZE(img1)
    kp2, desc2 = common.get_AKAZE(img2)
    matches = find_matches(desc1, desc2, ratioThreshold=0.73)  #get matches
    XY = common.get_match_points(kp1, kp2, matches)
    H = RANSAC_fit_homography(XY)                               #get H matrix using RANSAC
    corners1 = np.array([[0,0],[img1.shape[1]-1, 0], [img1.shape[1]-1, img1.shape[0]-1], [0, img1.shape[0]-1]])
    corners2 = np.array([[0,0],[img2.shape[1]-1, 0], [img2.shape[1]-1, img2.shape[0]-1], [0, img2.shape[0]-1]])
    Hinv = np.linalg.inv(H)
    Hinv = Hinv / Hinv[2, 2]
    im1T = homography_transform(corners1, np.array([[1,0,0],[0,1,0],[0,0,1]]))
    im2T = homography_transform(corners2, Hinv)
    cornersT = np.vstack((im1T, im2T))
    imgSize = (int(np.amax(cornersT[:,0], axis=0)-np.amin(cornersT[:,0], axis=0)),
               int(np.amax(cornersT[:,1], axis=0)-np.amin(cornersT[:,1], axis=0)))
    T = np.array([[1,0,-1*np.amin(cornersT[:,0], axis=0)],
                 [0,1, -1*np.amin(cornersT[:,1], axis=0)],
                 [0, 0, 1]])
    res1 = cv2.warpPerspective(np.ones((img2.shape[0], img2.shape[1])), (np.dot(T, Hinv)), imgSize)
    res2 = cv2.warpPerspective(np.ones((img1.shape[0], img1.shape[1])), np.dot(T, (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))), imgSize)
    res = np.add(res1, res2)
    print(res.shape)
    print(imgSize)
    resCol1 = cv2.warpPerspective(img2, (np.dot(T, Hinv)), imgSize)
    resCol2 = cv2.warpPerspective(img1, np.dot(T, (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))), imgSize)
    stitched = np.zeros((res.shape[0], res.shape[1], 3))

    for x in range(imgSize[1]):
        for y in range(imgSize[0]):
            if res[x, y] == 1:
                if res1[x, y] == 1 and res2[x, y] == 0:
                    stitched[x, y] = resCol1[x, y]
                if res1[x, y] == 0 and res2[x, y] == 1:
                    stitched[x, y] = resCol2[x, y]
            if res[x, y] == 2:
                stitched[x, y] = resCol1[x, y]*.5 + resCol2[x, y]*.5
    return stitched 



if __name__ == "__main__":

    #Possible starter code; you might want to loop over the task 6 images

    to_stitch = 'florence2'
    I1 = read_img(os.path.join('task6',to_stitch,'p1.jpg'))
    I2 = read_img(os.path.join('task6',to_stitch,'p2.jpg'))
    res = make_warped(I1,I2)
    save_img(res,"result_"+to_stitch+".jpg")


    to_stitch = 'lowetag'
    I1 = read_img(os.path.join('task6', to_stitch, 'p1.jpg'))
    I2 = read_img(os.path.join('task6', to_stitch, 'p2.jpg'))
    kp1, desc1 = common.get_AKAZE(I1)
    kp2, desc2 = common.get_AKAZE(I2)
    matches = find_matches(desc1, desc2, 0.73)
    res = draw_matches(I1, I2, kp1, kp2, matches)
    save_img(res, "result_sparse_" + to_stitch + ".jpg")

