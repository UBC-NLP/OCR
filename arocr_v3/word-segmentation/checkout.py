"""
For Testing purposes
    Take image from user, crop the background and transform perspective
    from the perspective detect the word and return the array of word's
    bounding boxes
"""

import page
import words
from PIL import Image
import cv2

# User input page image
print(cv2.imread("/Users/ahsanghani/Documents/NLP/OCR/OCR/arocr_v3/word-segmentation/test.jpg"))
print(cv2.imread("/Users/ahsanghani/Documents/NLP/OCR/OCR/arocr_v3/word-segmentation/test.jpg").shape)

image = cv2.cvtColor(cv2.imread("/Users/ahsanghani/Documents/NLP/OCR/OCR/arocr_v3/word-segmentation/test.jpg"), cv2.COLOR_BGR2RGB)

# Crop image and get bounding boxes
crop = page.detection(image)
print("1")
boxes = words.detection(crop)
print("2")
lines = words.sort_words(boxes)
print("done")
# Saving the bounded words from the page image in sorted way
for i, line in enumerate(lines):
    text = crop.copy()
    print(i)
    for (x1, y1, x2, y2) in line:
        roi = text[y1:y2, x1:x2]
        save = Image.fromarray(text[y1:y2, x1:x2])
        print(i)
        save.save(f"/Users/ahsanghani/Documents/NLP/OCR/OCR/arocr_v3/word-segmentation/segmented/segment" + str(i) + ".png")

