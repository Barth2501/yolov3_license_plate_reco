import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from .utils import *

def main(input_dir, output_dir, image, batch_detection = False):

    if batch_detection:
        images_names = os.listdir(input_dir)
        for image_name in images_names:
            if image_name[-3:]=='jpg':
                img = cv2.imread(os.path.join(input_dir, image_name),0)
                plt.imshow(img, cmap='gray')
                plt.title('Cropped image')
                plt.pause(1)
                # Rotation of the image
                r_img = rotate_img(img)

                # Expansion of the image
                exp_img = expansion_img(r_img)

                # Egalisation of the image
                eg_img = egalisation_img(exp_img)

                # Vertical cut
                cut_img = cropping_border(eg_img)

                # Binarization
                b_img = binarization(cut_img)

                # Vertical crop
                v_crop_img = vertical_crop(b_img)

                # Horizontal crop
                h_crop_img = horizontal_crop(v_crop_img,4)

                # Segmentation
                caracter_list_image = segmentation(h_crop_img)
                # Remove the noises from the cropped results
                caracter_list_image = remove_noises(caracter_list_image)
                for i, character in enumerate(caracter_list_image):

                    crop_char_and_save(character, name=os.path.join(output_dir, image_name.split('.')[0]) + '_char_' + str(i))
        
    else:
        if isinstance(image,str):
            image = cv2.imread(image,0)
        plt.imshow(image, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Rotation of the image
        r_img = rotate_img(image)
        plt.imshow(r_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Expansion of the image
        exp_img = expansion_img(r_img)
        plt.imshow(exp_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Egalisation of the image
        eg_img = egalisation_img(exp_img)
        plt.imshow(eg_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Vertical cut
        cut_img = cropping_border(eg_img)
        plt.imshow(cut_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Binarization
        b_img = binarization(cut_img)
        plt.imshow(b_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Vertical crop
        v_crop_img = vertical_crop(b_img)
        plt.imshow(v_crop_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Horizontal crop
        h_crop_img = horizontal_crop(v_crop_img,4)
        plt.imshow(h_crop_img, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        # Segmentation
        caracter_list_image = segmentation(h_crop_img)
        # Remove the noises from the cropped results
        caracter_list_image = remove_noises(caracter_list_image)
        characters = []
        for i, character in enumerate(caracter_list_image):
            character = crop_char_and_save(character)
            characters.append(character)
            plt.imshow(character, cmap='gray')
            plt.title('digit n°{}'.format(i))
            plt.pause(1)
        plt.close()
        
        return characters