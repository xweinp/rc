import cv2
import numpy as np

from calibrate import get_img_size


def get_undistort_maps(
    camera_matrix,
    dist_coeffs,
    img_size,
    alpha=0
):
    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        img_size, alpha
    )[0]
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3),
        rect_camera_matrix,
        img_size,
        cv2.CV_32FC1
    )
    return map1, map2


def undistort_image(img, map1, map2):
    img_undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return img_undist


def undistort_images(
    images,
    camera_matrix,
    dist_coeffs,
    alpha=0
):
    img_size = get_img_size(images[0])
    map1, map2 = get_undistort_maps(
        camera_matrix,
        dist_coeffs,
        img_size,
        alpha
    )
    undistorted_images = [
        undistort_image(img, map1, map2) for img in images
    ]
    return undistorted_images
