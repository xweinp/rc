import os
import cv2


def load_aruco_images(path="data/calibration/"):
    files = os.listdir(path)
    files = [f for f in files if f.startswith("aruco")]
    images = [cv2.imread(os.path.join(path, f)) for f in files]
    return images, files


def load_images_from_path(path):
    images = []
    files = os.listdir(path)
    images = [cv2.imread(os.path.join(path, f)) for f in files]
    return images, files


def save_image(image, path):
    cv2.imwrite(path, image)


def save_images(images, names, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for image, name in zip(images, names):
        filepath = os.path.join(directory, name)
        cv2.imwrite(filepath, image)
