import cv2
from imgaug import augmenters as iaa
import os

flip = iaa.Fliplr()
blur = iaa.GaussianBlur(sigma=(0.8,1.5))

images_train_dir = os.listdir('./train')

for i, image in enumerate(images_train_dir):
    img = cv2.imread('./train/' + image)
    print(i)
    flipped = flip.augment_image(img)
    blurred = blur.augment_image(img)
    flipped_blurred = blur.augment_image(flipped)
    cv2.imwrite('./train/flipped_' + image,flipped)
    cv2.imwrite('./train/blurred_' + image,blurred)
    cv2.imwrite('./train/flip+blur_' + image,flipped_blurred)