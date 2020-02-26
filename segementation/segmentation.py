import cv2
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from .utils import segment

def main(_argv):

    img = cv2.imread(FLAGS.image)
        
    caracter_list_image = segment(img, save=False)

    for i, character in enumerate(caracter_list_image):
        segment(character, save=True, name=str(FLAGS.image).split('.')[-2][1:]+str(i))

if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('image','../output_cropped.jpg',"Choose the image you want to segment")

    app.run(main)