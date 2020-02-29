import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from .models.yolo import (
    YoloV3
)
from .datasets.dataset import transform_images, load_tfrecord_dataset
from .models.utils import draw_outputs

# flags.DEFINE_boolean('batch_detection', False, 'False if you want to detect only one imgae')
# flags.DEFINE_string('image','','Image you want to predict, to use only if batch detection is disabled')
# flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
# flags.DEFINE_string('weights', './checkpoints/yolov3_train_8.tf',
#                     'path to weights file')
# flags.DEFINE_string('classes', './data/license.plate', 'path to classes file')
# flags.DEFINE_string('input_images', './data/jpg/test/', 'path to input images folder')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_string('output', './outputs/global_outputs/', 'path to output image')
# flags.DEFINE_string('cropped_output', './outputs/cropped_outputs/', 'path to output cropped image')


def main(classes, weights, output, cropped_output, batch_detection=False, input_images=None, image=None, size=416, num_classes=80):

    yolo = YoloV3(classes = num_classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    if batch_detection:
        images_names = os.listdir(input_images)

        for image_name in images_names:
            img_raw = tf.image.decode_image(
                open(os.path.join(input_images, image_name), 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)

            img = transform_images(img, size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))

            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            img, cropped_images = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(os.path.join(output, image_name.split('.')[0]) + '.jpg', img)
            for i, cropped_image in enumerate(cropped_images):
                cv2.imwrite(os.path.join(cropped_output, image_name.split('.')[0]) + '_' + str(i) + '.jpg', cropped_image)
            logging.info('output saved to: {}'.format(os.path.join(output, image_name.split('.')[0]) + '.jpg'))
    
    else:
        img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img, cropped_images = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(os.path.join(output, image.split('.')[0]) + '.jpg', img)
        for i,cropped_image in enumerate(cropped_images):
            cv2.imwrite(os.path.join(cropped_output, image.split('.')[0]) + '_' + str(i) + '.jpg', cropped_image)
        return cropped_image

# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass