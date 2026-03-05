import cv2
import numpy as np


# helpful guide
# https://medium.com/thedeephub/detecting-and-tracking-objects-with-orb-using-opencv-d228f4c9054e
class ORBDetector:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=5, patchSize=32)
        self.matcher = cv2.BFMatcher(crossCheck=True)

    def find_matches(self, img1: np.ndarray, img2: np.ndarray):
        keypoints_1, descriptors_1 = self.orb.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = self.orb.detectAndCompute(img2, None)
        matches = self.matcher.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        return keypoints_1, keypoints_2, matches

    def find_homographies(self, image_pairs):
        Hs = []
        for img1, img2 in image_pairs:
            keypoints_1, keypoints_2, matches = self.find_matches(img1, img2)

            points_1 = np.array(
                [keypoints_1[match.queryIdx].pt for match in matches], dtype=np.float32
            )
            points_2 = np.array(
                [keypoints_2[match.trainIdx].pt for match in matches], dtype=np.float32
            )

            H, mask = cv2.findHomography(
                points_1, points_2, cv2.RANSAC, 5, maxIters=100000
            )
            Hs.append(H)
        return Hs
