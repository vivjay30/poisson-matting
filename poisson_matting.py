from __future__ import division

import numpy as np
import scipy.ndimage
from imageio import imread
import cv2
from numba import jit


BG_SIGMA = 2.0/256  # Amount of variance modeling the noise of the known background
MAX_COLOR_DIST = 100  # Clamb higher color distances in CIELAB
MIN_COLOR_DIST = 20  # Threshold for foreground in BG Sub


def segmentation_to_trimap(segmentation, K1=5, K2=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.greater(segmentation, 250).astype(np.float32))
    fg = cv2.erode(fg, kernel, iterations=K1)
    unknown = np.array(np.not_equal(segmentation, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=K2)
    trimap = fg * 255 + (unknown - fg) * 128

    # plt.title("Trimap")
    # plt.imshow(trimap, cmap='gray')
    # plt.show()
    return trimap


def generate_trimap(segmentation, image, gt_background, K1=15, K2=15):
    """Generates Trimap from segmentation and background"""
    base_trimap = segmentation_to_trimap(segmentation, K1, K2)
    generous_trimap = segmentation_to_trimap(segmentation, 3*K1, 3*K2)

    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gt_bg_lab = cv2.cvtColor(gt_background, cv2.COLOR_RGB2LAB)

    diff = img_lab.astype(np.float32) - gt_bg_lab.astype(np.float32)
    distance = np.minimum(np.linalg.norm(diff, ord=2, axis=2), MAX_COLOR_DIST)
    distance = cv2.GaussianBlur(np.expand_dims(distance, 2),(3,3), 0)

    fg = distance > MIN_COLOR_DIST

    # plt.title("Foreground")
    # plt.imshow(fg, cmap='gray')
    # plt.show()

    # Find where bg sub disagrees with normal trimap
    false_foreground = np.logical_and(base_trimap == 255, ~fg)  # Trimap says fg, bg sub says bg
    false_background = np.logical_and(base_trimap == 0, fg)  # Trimap says bg, bg sub says fg

    kernel = np.ones((3,3),np.uint8)
    false_foreground = np.expand_dims(false_foreground, 2).astype(np.uint8) * 255
    false_foreground = cv2.dilate(false_foreground, kernel, iterations = 1)
    # false_foreground = cv2.morphologyEx(false_foreground, cv2.MORPH_CLOSE, kernel)
    # false_foreground = cv2.morphologyEx(false_foreground, cv2.MORPH_OPEN, kernel)

    # plt.title("False Foreground")
    # plt.imshow(false_foreground, cmap='gray')
    # plt.show()

    # Take only false background in the generous trimap, but all false foreground
    potential_errors = np.logical_and(false_background, generous_trimap == 128)
    potential_errors = np.logical_or(potential_errors, false_foreground)
    trimap = base_trimap * (1 - potential_errors) + 128 * potential_errors

    # plt.title("Trimap")
    # plt.imshow(trimap, cmap='gray')
    # plt.show()

    return base_trimap, trimap.astype(np.uint8)


@jit
def computeAlphaJit(alpha, b, unknown):
    h, w = alpha.shape
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    while (n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold):
        alphaOld = alphaNew.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                if(unknown[i,j]):
                    alphaNew[i,j] = 1/4 * (alphaNew[i-1 ,j] + alphaNew[i,j-1] + alphaOld[i, j+1] + alphaOld[i+1,j] - b[i,j])
        n +=1
    return alphaNew


def poisson_matte(gray_img, trimap, gt_background_gray):
    h, w = gray_img.shape

    gray_img = gray_img/255.
    gt_background_gray = gt_background_gray / 255.

    fg = trimap == 255
    bg = trimap == 0

    for i in range(1):

        unknown = True ^ np.logical_or(fg,bg)
        fg_img = gray_img*fg
        bg_img = gray_img*bg
        alphaEstimate = fg + 0.5 * unknown

        plt.title("Alpha Estimate")
        plt.imshow(alphaEstimate, cmap='gray')
        plt.show()

        approx_bg = gt_background_gray
        #approx_bg = cv2.inpaint((bg_img * 255).astype(np.uint8),(unknown +fg ).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(fg)).astype(np.float32) / 255.
        approx_fg = cv2.inpaint((fg_img * 255).astype(np.uint8),(unknown +bg ).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(bg)).astype(np.float32) / 255.

        # Smooth F - B image
        approx_diff = approx_fg - approx_bg
        approx_diff = scipy.ndimage.filters.gaussian_filter(approx_diff, 0.9)

        dy, dx = np.gradient(gray_img)
        d2y, _ = np.gradient(dy/approx_diff)
        _, d2x = np.gradient(dx/approx_diff)
        b = d2y + d2x

        alpha = computeAlphaJit(alphaEstimate, b, unknown)
        
        alpha = np.minimum(np.maximum(alpha,0),1).reshape(h,w)

        plt.title("Alpha")
        plt.imshow(alpha, cmap='gray')
        plt.show()

        fg = np.logical_or(trimap == 255, alpha > 0.95)
        bg = np.logical_or(trimap == 0, alpha < 0.05)


    return alpha

# Load in image
def main():
    base_filename = "xuan_kitchen_9"
    img = cv2.imread("{}_img.png".format(base_filename))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gt_background = cv2.imread("{}_back.png".format(base_filename))
    gt_background_gray = cv2.cvtColor(gt_background, cv2.COLOR_BGR2GRAY).astype(np.float32)

    segmentation = cv2.imread("{}_masksDL.png".format(base_filename), 0)
    base_trimap, trimap = generate_trimap(segmentation, img, gt_background)

    # img = cv2.imread('troll.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # gray_img = cv2.imread('troll.png', 0).astype(np.float32)
    # trimap = cv2.imread('trollTrimap.bmp', 0)
    # scale = 0.1
    # img = scipy.misc.imresize(img, scale)
    # gray_img = scipy.misc.imresize(gray_img, scale)
    # trimap = scipy.misc.imresize(trimap, scale)

    alpha = poisson_matte(gray_img, trimap, gt_background_gray)

    plt.title("Alpha Out")
    plt.imshow(alpha, cmap='gray')
    plt.show()
    h, w, c = img.shape

    plt.title("Matte Out")
    plt.imshow((alpha.reshape(h,w,1).repeat(3,2)*img).astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()