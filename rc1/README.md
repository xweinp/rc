### Jakub Pniewski 459481

# Robot Control - Homework 1

## Preparation

First you should create a virtual environment using uv and install all required packages:

```bash
uv sync
```

To run the code in the notebooks you will have to select virtual environment created by uv. It should be in the `rc1/.venv` folder.

## Task 1 - Camera Calibration and Image Undistortion

### 1. Calibrate the camera

I decided to use ArUco markers for calibration. They are easy to detect and provide good accuracy. I use whole ChArUco boards.

I also tried using the dots, but the results were not very accurate. I had to tweak a lot of parameters to even detect the dots. 

OpenCV has functions that we can use for the ChArUco board detection and calibration. The code I've written is overall quite short and simple. It is available in the `calibration/calibrate.py` file.

### 2. Visualize the detected patterns

On vast majority of photos all corners and markers are detected correctly. There are some photos of lower quality. On them some points are missed. However, forcing the detection of all points would probably just introduce noise and make the calibration worse.

Functions I use for visualization are in the `utils/plot.py` file.

#### Here are some examples:

| All Corners Detected | Some Corners Missed |
| :---: | :---: |
| ![All corners detected](data/plots/charuco_detected_corners_1.png) | ![Some corners missed](data/plots/charuco_detected_corners_2.png) |

| ArUco Markers Detected | Some Markers Missed |
| :---: | :---: |
| ![ArUco markers detected](data/plots/charuco_detected_tags_1.png) | ![Some markers missed](data/plots/charuco_detected_tags_2.png) |


All visualizations are available in the `calibration/calibration.ipynb` notebook.

### 3. Undistort the stitching photos

The images were undistorted with standard OpenCV functions. The code is available in the `calibration/undistort.py` file.

#### The results can be seen below:

| Original image | Undistorted image |
| :---: | :---: |
| ![](data/images/set_1_1.jpg) | ![](data/undistorted/set_1_1.jpg) |
| ![](data/images/set_1_2.jpg) | ![](data/undistorted/set_1_2.jpg) |
| ![](data/images/set_2_1.jpg) | ![](data/undistorted/set_2_1.jpg) |
| ![](data/images/set_2_2.jpg) | ![](data/undistorted/set_2_2.jpg) |
| ![](data/images/set_3_1.jpg) | ![](data/undistorted/set_3_1.jpg) |
| ![](data/images/set_3_2.jpg) | ![](data/undistorted/set_3_2.jpg) |

### 4. Document your process: 

I did some experiments with different strategies:

- using the dots
- using the ArUco markers only (from images with ChArUco boards)
- using the ChArUco board

The last option proved to be the best. The dots were hard to detect. Even with some tuning, the detection was not very reliable. Using only the ArUco markers means losing some information compared to using the full ChArUco board. I used it mainly because I didn't realize that extra information about the size of the black squares on board was sent on slack.

The code used for calibration is pretty straightforward. It can be seen in the `calibration/calibrate.py` file. There were some challenges due to OpenCV's API and poor documentation. For example, without `board.setLegacyPattern(True)` the calibration would just not work properly. It's only mentioned in some random place in the docs.

Undistortion was really easy as I have already done it before. I decided to go with alpha equal to 0 in `cv2.getOptimalNewCameraMatrix` (so the image is cropped to a rectangle). This way we loose some pixels from the corners, but I don't have to care about the black areas on the sides of the image during stitching.

The whole code for undistortion is in the `calibration/undistort.py` file.

Usage of the code is documented in the `calibration/calibration.ipynb` notebook.


### 5. Metrics

The main metric I used to evaluate the results is RMS of reprojection error. The final RMS error I achieved is about **0.32345** pixels. This is a pretty good result.

I also wanted to see how the number of images used for calibration affects the error. The results can be seen in the following plot:

| Whole  | Zoomed in |
| :---: | :---: |
| ![](data/plots/n_imgs_vs_rms.png) | ![](data/plots/n_imgs_vs_rms_zoomed.png) |

I tried it with shuffling the images a few times. The results were similar. Generally, using more than 4 images doesn't improve the results significantly. The error stabilizes around ~0.32 pixels. Without verification on unseen images, we would not know that for using just one image the error is actually much higher than what the error on it might suggest.

From `cv2.calibrateCameraExtended` we can also get some additional metrics:
- standard deviation for each intrinsic parameter (in order $[f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, \tau_x, \tau_y]$.)
- standard deviation for each extrinsic parameter (in order $[r_x, r_y, r_z, t_x, t_y, t_z]$)
- per view reprojection errors

The values suggest that the calibration went well. I've plotted them and the results can be seen below:

| Standard Deviation of Intrinsic Parameters | Standard Deviation of Extrinsic Parameters | Per View Reprojection Errors |
| :---: | :---: | :---: |
| ![](data/plots/intrinsic_std.png) | ![](data/plots/extrinsic_std.png) | ![](data/plots/per_view_repr.png) |


## Task 2 - Projective Transformation

I've implemented the projective transformation. The code is in `projective_transformation/transformation.py` file:
- `project_on_plane`: projects an image on a given plane (for example on another image)
- `project_transform`: projects an image on an empty plane according to the homography

I've checked it in the `projective_transformation/projection.ipynb` notebook. I've used an exmple homography matrix:

$$
H = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0.0005 & 0 & 1
\end{bmatrix}
$$



| Original Image | Projected image| Inverse Projected Image |
| :---: | :---: | :---: |
| ![](data/undistorted/set_3_1.jpg) | ![](data/plots/image_transformed.png) | ![](data/plots/image_restored.png) |  

We can see that the inverse projection correctly restores the original image. We back-projected the balck pixels as well so it may look weird in markdown preview.

## Task 3 - Finding Projective Transformation

Function `find_transformation` is implemented in the `projective_transformation/transformation.py` file. It implements the method described during the lecture. First I construct matrix $A$ such that for $\mathbf{h} = [h_1, h_2, \dots, h_9]^T$ there is $Ah = 0$. Then I use SVD to find the solution. It is tested by a fucntion in the `projective_transformation/transformation_test.py` file. You can run the tests either from the notebook `projective_transformation/projection.ipynb` or with command:

```bash
uv run python ./projective_transformation/transformation_test.py
```

The tests confirm that the implementation is correct.

## Task 4 - Finding Projective Transformation by Hand

I selected some pairs od points for every pair of images. I tried to be as accurate as possible. The images of selected points are in the `data/selected_points` folder. The file `set_a_b_c.png` means that this is a `c`-th point for pair of images `a` taken from image `b`.

Examples:

| Image 1 | Image 2 |
| :---: | :---: |
| ![](data/selected_points/set_1_1_1.png) | ![](data/selected_points/set_1_2_1.png) |
| ![](data/selected_points/set_2_1_3.png) | ![](data/selected_points/set_2_2_3.png) |
| ![](data/selected_points/set_3_1_2.png) | ![](data/selected_points/set_3_2_2.png) |

I find the transformation matrices in `transformation_by_hand.ipynb` notebook.

After performing the stitching I had to find some new pairs and get rid of some bad ones. Originally I wanted to use 4 points per pair. The pairs I use in the end are saved in the `data/selected_points/point_pairs.csv` file.

## Task 5 - Image Stitching

First I perform the simple projection using the homography matrices found in Task 4. The results can be seen below:

|Image 1| Image 2 | Simple Projection |
| :---: | :---: | :---: |
| ![](data/undistorted/set_1_1.jpg) | ![](data/undistorted/set_1_2.jpg) | ![](data/hand_selected_points/simple_projection/pair_1.png) |
| ![](data/undistorted/set_2_1.jpg) | ![](data/undistorted/set_2_2.jpg) | ![](data/hand_selected_points/simple_projection/pair_2.png) |
| ![](data/undistorted/set_3_1.jpg) | ![](data/undistorted/set_3_2.jpg) | ![](data/hand_selected_points/simple_projection/pair_3.png) |

We can see that the edges fit but are not perfect. There are some misalignments. However, this way of stitching assumes that the images are taken from the same point of view. And this does not really hold in our case. Also for pair 2 it is hard to find a lot of corresponding points due to small overlap area.

I've also implemented linear blending to improve the results. The results can be seen below:

|Image 1| Image 2 | Linear Blending |
| :---: | :---: | :---: |
| ![](data/undistorted/set_1_1.jpg) | ![](data/undistorted/set_1_2.jpg) | ![](data/hand_selected_points/linear_stitching/pair_1.png) |
| ![](data/undistorted/set_2_1.jpg) | ![](data/undistorted/set_2_2.jpg) | ![](data/hand_selected_points/linear_stitching/pair_2.png) |
| ![](data/undistorted/set_3_1.jpg) | ![](data/undistorted/set_3_2.jpg) | ![](data/hand_selected_points/linear_stitching/pair_3.png) |

On pair 3 we can see it does a very good job. On pair 2 which is not aligned so well it gives some improvement in the middle but some artifacts further away from the center. On image 2 we get a lot of ghosting artifacts, because the photos have a very big overlap and I calculate distance form the nearest edge. 

Code for stitching is in the `projective_transformation/transformation.py` file. I find homography matrices and then use the reverse projection to project pixels from second image to the first one. The example usage is in the `projective_transformation/transformation_by_hand.ipynb` notebook.

Panoramas are saved in `data/results` folder.

## Task 6 - Robust Image Stitching with ORB and RANSAC

The ORB detector is implemented in the `projective_transformation/orb.py` file. It uses OpenCV's implementation of ORB and BFMatcher to find matching keypoints between two images. The homography is then estimated using RANSAC to be robust to outliers.

The important part was to set `crossCheck=True` in the BFMatcher. It limits the amount of outliers significantly. I also had to tweak other parameters - especially the `nfeatures` in the ORB detector. If it is too low, not enough keypoints are found. If it is too high, too many outliers are present. I also increased the number of iterations in RANSAC.

The results of simple projection can be seen below:

|Image 1| Image 2 | Simple Projection with ORB and RANSAC |
| :---: | :---: | :---: |
| ![](data/undistorted/set_1_1.jpg) | ![](data/undistorted/set_1_2.jpg) | ![](data/orb/simple_projection/pair_1.png) |
| ![](data/undistorted/set_2_1.jpg) | ![](data/undistorted/set_2_2.jpg) | ![](data/orb/simple_projection/pair_2.png) |
| ![](data/undistorted/set_3_1.jpg) | ![](data/undistorted/set_3_2.jpg) | ![](data/orb/simple_projection/pair_3.png) |

In these case pairs 1 and 3 are pretty simmilar to the ones obtained in the previous task. Pair 2 is a bit worse. It is hard to find good keypoints there due to small overlap area.

I also performed linear blending to improve the results:

|Image 1| Image 2 | Linear Blending with ORB and RANSAC |
| :---: | :---: | :---: |
| ![](data/undistorted/set_1_1.jpg) | ![](data/undistorted/set_1_2.jpg) | ![](data/orb/linear_stitching/pair_1.png) |
| ![](data/undistorted/set_2_1.jpg) | ![](data/undistorted/set_2_2.jpg) | ![](data/orb/linear_stitching/pair_2.png) |
| ![](data/undistorted/set_3_1.jpg) | ![](data/undistorted/set_3_2.jpg) | ![](data/orb/linear_stitching/pair_3.png) |

All the images are available in the `data/results` folder.