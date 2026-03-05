import cv2
import matplotlib.pylab as plt
import numpy as np

PLT_FIGSIZE = (8, 6)


def plot_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=PLT_FIGSIZE, frameon=False)
    fig.patch.set_alpha(0)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(img)
    # Turn off white borders
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.show()


def plot_two_imgs(img1, img2, titles=("", "")):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=PLT_FIGSIZE)
    fig.patch.set_alpha(0)

    for ax, im, ttl in zip(axs, (img1, img2), titles):
        ax.imshow(im)
        ax.axis("off")
        if ttl:
            ax.set_title(ttl)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    plt.margins(0, 0)
    plt.show()


def plot_tags(img, corners, ids):
    img_draw = img.copy()
    cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)
    plot_img(img_draw)


def plot_charuco(img, corners, ids):
    img_draw = img.copy()
    cv2.aruco.drawDetectedCornersCharuco(img_draw, corners, ids)
    plot_img(img_draw)


def plot_metric(metric, title, xlabel, ylabel):
    plt.figure(figsize=PLT_FIGSIZE)
    plt.bar(np.arange(len(metric)), metric[:, 0])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
