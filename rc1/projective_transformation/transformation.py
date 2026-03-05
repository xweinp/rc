import numpy as np


def get_transformed_bbox(w: int, h: int, H: np.ndarray) -> np.ndarray:
    inf = 1e9
    y_min, y_max = inf, -inf
    x_min, x_max = inf, -inf
    for y in [0, h - 1]:
        for x in [0, w - 1]:
            coords = np.array([x, y, 1]).T
            src_coords = H @ coords
            src_coords /= src_coords[2]
            if src_coords[2] <= 1e-6:
                continue
            x_, y_ = int(round(src_coords[0])), int(round(src_coords[1]))
            x_min = min(x_min, x_)
            x_max = max(x_max, x_)
            y_min = min(y_min, y_)
            y_max = max(y_max, y_)
    return np.array([x_min, y_min, x_max, y_max])


def _transform_pixels(
    img: np.ndarray,
    H_inv: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    res: np.ndarray,
):
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            coords = np.array([x, y, 1]).T
            src_coords = H_inv @ coords
            src_coords /= src_coords[2]

            src_x, src_y = int(round(src_coords[0])), int(round(src_coords[1]))
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                res[y - y_min, x - x_min] = img[src_y, src_x]


# img -H-> plane
def project_on_plane(img: np.ndarray, H: np.ndarray, plane: np.ndarray):
    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]

    (x_min, y_min, x_max, y_max) = get_transformed_bbox(img.shape[1], img.shape[0], H)

    x_min = min(x_min, 0)
    y_min = min(y_min, 0)
    x_max = max(x_max, plane.shape[1] - 1)
    y_max = max(y_max, plane.shape[0] - 1)

    res = np.zeros(
        (y_max - y_min + 1, x_max - x_min + 1, img.shape[2]), dtype=img.dtype
    )
    res[-y_min : plane.shape[0] - y_min, -x_min : plane.shape[1] - x_min] = plane

    _transform_pixels(img, H_inv, x_min, y_min, x_max, y_max, res)

    return res


# img -H-> empty plane
def project_transform(img: np.ndarray, H: np.ndarray):
    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]

    (x_min, y_min, x_max, y_max) = get_transformed_bbox(img.shape[1], img.shape[0], H)
    res = np.zeros(
        (y_max - y_min + 1, x_max - x_min + 1, img.shape[2]), dtype=img.dtype
    )

    _transform_pixels(img, H_inv, x_min, y_min, x_max, y_max, res)

    return res


# [x y] -> [x y 1]
def add_ones(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)


def find_transformation(points1: np.ndarray, points2: np.ndarray):
    n = points1.shape[0]

    # points2 = H * points1
    # find H

    A = np.zeros((2 * points1.shape[0], 9), dtype=np.float32)
    for i in range(n):
        A[2 * i, :2] = points1[i]
        A[2 * i, 2] = 1
        A[2 * i + 1, 3:5] = points1[i]
        A[2 * i + 1, 5] = 1

        A[2 * i, 6:9] = -points2[i, 0]
        A[2 * i + 1, 6:9] = -points2[i, 1]

        A[2 * i, 6:8] *= points1[i]
        A[2 * i + 1, 6:8] *= points1[i]

    _, _, V = np.linalg.svd(A)
    eingenvector = V[-1, :]
    return eingenvector.reshape(3, 3) / eingenvector[-1]


def _stitch_pixels(
    img: np.ndarray,
    res: np.ndarray,
    H_inv: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    bbox1: np.ndarray,
    bbox2: np.ndarray,
):
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            coords = np.array([x, y, 1]).T
            src_coords = H_inv @ coords
            src_coords /= src_coords[2]

            src_x, src_y = int(round(src_coords[0])), int(round(src_coords[1]))
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                if res[y - y_min, x - x_min].sum() == 0:
                    res[y - y_min, x - x_min] = img[src_y, src_x]
                else:
                    dist_1 = (
                        min(
                            abs(src_x - bbox1[0]),
                            abs(src_y - bbox1[1]),
                            abs(src_x - bbox1[2]),
                            abs(src_y - bbox1[3]),
                        )
                        + 1e-6
                    )
                    dist_2 = (
                        min(
                            abs(x - x_min - bbox2[0]),
                            abs(y - y_min - bbox2[1]),
                            abs(x - x_min - bbox2[2]),
                            abs(y - y_min - bbox2[3]),
                        )
                        + 1e-6
                    )
                    alpha = dist_1 / (dist_1 + dist_2)
                    res[y - y_min, x - x_min] = (
                        res[y - y_min, x - x_min].astype(np.float32) * (1 - alpha)
                        + img[src_y, src_x].astype(np.float32) * alpha
                    ).astype(img.dtype)


def linear_stitching(img: np.ndarray, H: np.ndarray, plane: np.ndarray):
    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]

    (x_min, y_min, x_max, y_max) = get_transformed_bbox(img.shape[1], img.shape[0], H)
    x_min = min(x_min, 0)
    y_min = min(y_min, 0)
    x_max = max(x_max, plane.shape[1] - 1)
    y_max = max(y_max, plane.shape[0] - 1)

    res = np.zeros(
        (y_max - y_min + 1, x_max - x_min + 1, img.shape[2]), dtype=img.dtype
    )
    res[-y_min : plane.shape[0] - y_min, -x_min : plane.shape[1] - x_min] = plane
    bbox1 = np.array([0, 0, img.shape[1] - 1, img.shape[0] - 1])
    bbox2 = np.array([0, 0, plane.shape[1] - 1, plane.shape[0] - 1])
    _stitch_pixels(img, res, H_inv, x_min, y_min, x_max, y_max, bbox1, bbox2)

    return res
