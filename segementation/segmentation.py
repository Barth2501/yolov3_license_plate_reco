import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from .utils import segment, crop_char_and_save

def main(input_dir, output_dir, image, batch_detection = False):

    if batch_detection:
        images_names = os.listdir(input_dir)
        for image_name in images_names:

            img = cv2.imread(os.path.join(input_dir, image_name))
                
            caracter_list_image = segment(img, save=False)

            for i, character in enumerate(caracter_list_image):

                crop_char_and_save(character, name=os.path.join(output_dir, image_name.split('.')[0]) + '_char_' + str(i))
    
    else:
        plt.imshow(image, cmap='gray')
        plt.title('Cropped image')
        plt.pause(1)
        caracter_list_image = segment(image, save=False)
        characters = []
        for i, character in enumerate(caracter_list_image):
            character = crop_char_and_save(character)
            characters.append(character)
            plt.imshow(character, cmap='gray')
            plt.title('digit n°{}'.format(i))
            plt.pause(1)
        plt.close()
        
        return characters

# if __name__ == '__main__':

#     FLAGS = flags.FLAGS

#     flags.DEFINE_string('input_dir','../license_plate_detection/outputs/cropped_outputs',"Path of the input directory")
#     flags.DEFINE_string('output_dir','../character_reco/data/jpg','Path of the output directory')

#     app.run(main)