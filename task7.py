"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import homography_transform, RANSAC_fit_homography
from task6 import find_matches
import cv2
import os

def task7_warp_and_combine(scene, template, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.
    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - scene: Input image 1 of shape (H1,W1,3)
            template: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them
    Output - V: stitched image of size (?,?,3); unknown since it depends on H
                but make sure in V, for pixels covered by both scene and warped template,
                you see only template
    '''
    V = None
    return V

def improve_image(scene, template, transfer):
    '''
    Detect template image in the scene image and replace it with transfer image.

    Input - scene: image (H,W,3)
            template: image (K,K,3)
            transfer: image (L,L,3)
    Output - augment: the image with 
    
    Hints:
    a) You may assume that the template and transfer are both squares.
    b) This will work better if you find a nearest neighbor for every template
       keypoint as opposed to the opposite, but be careful about directions of the
       estimated homography and warping!
    '''
    kp1, desc1 = common.get_AKAZE(template)
    kp2, desc2 = common.get_AKAZE(scene)
    matches = find_matches(desc1, desc2, ratioThreshold=0.73)  # get matches
    XY = common.get_match_points(kp1, kp2, matches)
    H = RANSAC_fit_homography(XY)  # get H matrix using RANSAC
    transfer=cv2.resize(transfer, (template.shape[0], template.shape[1]))
    mask = cv2.warpPerspective(transfer, H, (scene.shape[1], scene.shape[0]))
    oneS = np.ones((scene.shape[0], scene.shape[1]))
    oneT = cv2.warpPerspective(np.ones((transfer.shape[0], transfer.shape[1])), H, (scene.shape[1], scene.shape[0]))
    res = np.add(oneS, oneT)
    print(res.shape)
    augment = np.zeros((res.shape[0], res.shape[1], 3))

    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            if res[x, y] == 1:
                augment[x, y] = scene[x, y]
            if res[x, y] == 2:
                augment[x, y] = mask[x, y]
    return augment

if __name__ == "__main__":
    # Task 7
    to_stitch = 'florence'
    s = read_img(os.path.join('task7', 'scenes', to_stitch, 'scene.jpg'))
    temp = read_img(os.path.join('task7', 'scenes', to_stitch, 'template.png'))
    trans = read_img(os.path.join('task7', 'scenes', 'bbb',  'template.png'))
    res = improve_image(s, temp, trans)
    save_img(res, "result_" + to_stitch + ".jpg")
    pass
