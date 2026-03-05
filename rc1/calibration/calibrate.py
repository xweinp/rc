import cv2
import numpy as np

from utils import get_img_size

ARUCO_DICT = cv2.aruco.DICT_4X4_1000
SQUARES_H = 16
SQUARES_W = 22
SQUARE_MM = 30
MARKER_MM = 22


def detect_charuco(images):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    charucoParams = cv2.aruco.CharucoParameters()

    detectorParams = cv2.aruco.DetectorParameters()
    detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    board = cv2.aruco.CharucoBoard(
        (SQUARES_W, SQUARES_H),
        SQUARE_MM,
        MARKER_MM,
        dictionary
    )
    board.setLegacyPattern(True)
    detector = cv2.aruco.CharucoDetector(board, charucoParams, detectorParams)

    all_charuco_corners = []
    all_charuco_ids = []
    all_object_points = []
    all_image_points = []
    all_marker_corners = []
    all_marker_ids = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray)
        if charuco_corners is None:
            continue

        object_points, image_points = board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        if object_points is None:
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        all_object_points.append(object_points)
        all_image_points.append(image_points)
        all_marker_corners.append(marker_corners)
        all_marker_ids.append(marker_ids)

    return (
        all_charuco_corners,
        all_charuco_ids,
        all_object_points,
        all_image_points,
        all_marker_corners,
        all_marker_ids
    )


def calibrate_charuco(images):
    _, _, object_points, image_points, _, _ = detect_charuco(images)
    img_size = get_img_size(images[0])

    (error,
     camera_matrix,
     dist_coeffs,
     rvecs,
     tvecs
     ) = cv2.calibrateCamera(
        object_points,
        image_points,
        img_size,
        None,
        None
    )
    return error, camera_matrix, dist_coeffs, rvecs, tvecs


def rms_reprojection_error(
    images,
    camera_matrix,
    dist_coeffs
):
    _, _, object_points, image_points, _, _ = detect_charuco(images)

    rvecs, tvecs = [], []
    for i in range(len(object_points)):
        _, rvec, tvec = cv2.solvePnP(
            object_points[i],
            image_points[i],
            camera_matrix,
            dist_coeffs
        )
        rvecs.append(rvec)
        tvecs.append(tvec)

    error = 0
    n_points = 0
    for i in range(len(object_points)):
        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        e = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)
        error += e**2
        n_points += len(object_points[i])
    rms_error = np.sqrt(error / n_points)

    return rms_error
