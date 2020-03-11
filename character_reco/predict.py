import os   
import numpy as np 
import tensorflow as tf
import cv2

from absl import flags, logging, app
from absl.flags import FLAGS
from .datasets import dataset as dataset


def main(image, size=28, batch_detection=False):

    model = tf.keras.models.load_model('./character_reco/models/CNN_saved_final_model')
    if batch_detection:
        img_raw = tf.image.decode_image(
            open(image, 'rb').read(), channels=1)
    else:
        img_raw = image
        img_raw = np.expand_dims(img_raw,axis=2)
    img = tf.expand_dims(img_raw,0)
    img = dataset.transform_images(img, size)

    prediction = model.predict(img)

    return np.argmax(prediction)