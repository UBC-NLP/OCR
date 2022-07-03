import numpy as np
import cv2 as cv
from preprocessing import binary_otsus, deskew
from utilities import projection, save_image
from glob import glob


def preprocess(image):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = cv.bitwise_not(gray_img)
    binary_img = binary_otsus(gray_img, 0)
    deskewed_img = deskew(binary_img)
    return deskewed_img


def projection_segmentation(clean_img, axis, cut=3):

    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    segments.append(clean_img[max(start-1, 0):idx, :])
                elif axis == 'vertical':
                    segments.append(clean_img[:, max(start-1, 0):idx])
                cnt = 0
                start = -1
    
    return segments

def line_horizontal_projection(image, cut=3):
    clean_img = preprocess(image)
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)
    return lines

def word_vertical_projection(line_image, cut=3):
    line_words = projection_segmentation(line_image, axis='vertical', cut=cut)
    line_words.reverse()
    return line_words


def extract_words(img, visual=0):
    lines = line_horizontal_projection(img)
    words = []
    for idx, line in enumerate(lines):
        if visual:
            save_image(line, 'lines', f'line{idx}')
        line_words = word_vertical_projection(line)
        for w in line_words:
            words.append((w, line))
    if visual:
        for idx, word in enumerate(words):
            save_image(word[0], 'words', f'word{idx}')
    return words


if __name__ == "__main__":
    
    img = cv.imread('../Dataset/scanned/capr196.png')
    extract_words(img, 1)