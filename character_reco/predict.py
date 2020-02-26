import os   
import numpy as np 
import tensorflow as tf
import cv2

from absl import flags, logging, app
from absl.flags import FLAGS
import datasets.dataset as dataset


FLAGS = flags.FLAGS

flags.DEFINE_string('image','./data/jpg/output_output_cropped0.jpg','Image you want to predict')
flags.DEFINE_integer('size',28,'Size of the image')

def main(_argv):

    model = tf.keras.models.load_model('./CNN_saved_final_model')
    img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=1)
    img = tf.expand_dims(img_raw,0)
    img = dataset.transform_images(img, FLAGS.size)
    prediction = model.predict(img)
    print(prediction)

if __name__ == "__main__":
    app.run(main)
