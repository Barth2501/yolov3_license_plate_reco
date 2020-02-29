from license_plate_detection.detect import main as license_plate_detection
from segementation.segmentation import main as segementation
from character_reco.predict import main as character_reco
from license_plate_detection.train import main as yolo_train
from character_reco.train import main as cnn_train
from absl import app, flags, logging


FLAGS = flags.FLAGS

flags.DEFINE_string('mode','predict', 'mode you want to use')
flags.DEFINE_boolean('batch_detection', False, 'False if you want to detect only one imgae')
flags.DEFINE_string('image','','Image you want to predict, to use only if batch detection is disabled')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('yolo_pretrained_weights', './license_plate_detection/datasets/yolo/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('yolo_weights', './license_plate_detection/checkpoints/yolov3_train_8.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './license_plate_detection/data/license.plate', 'path to classes file')
flags.DEFINE_string('input_images', './license_plate_detection/data/jpg/test/', 'path to input images folder')
flags.DEFINE_integer('yolo_size', 416, 'resize images to')
flags.DEFINE_string('yolo_output', './license_plate_detection/outputs/global_outputs/', 'path to output image')
flags.DEFINE_string('yolo_cropped_output', './license_plate_detection/outputs/cropped_outputs/', 'path to output cropped image')
flags.DEFINE_string('seg_output','./character_reco/data/jpg','dir where you want to save the segmented characters')
flags.DEFINE_string('yolo_train_dataset', './license_plate_detection/data/tfrecord/train_license_plate.tfrecord', 'path to train dataset as tfrecord')
flags.DEFINE_string('yolo_val_dataset', './license_plate_detection/data/tfrecord/valid_license_plate.tfrecord', 'path to validation dataset')
flags.DEFINE_integer('yolo_batch_size', 8, 'Yolo model batch size')
flags.DEFINE_float('yolo_learning_rate', 1e-3, 'learning rate of yolo model')
flags.DEFINE_integer('yolo_epochs', 8,'number of epochs of yolo model')

flags.DEFINE_string('cnn_train_dataset','./character_reco/data/tfrecord/train_character.tfrecord','Path to the train tfrecord dataset')
flags.DEFINE_string('cnn_val_dataset','./character_reco/data/tfrecord/val_character.tfrecord','Path to the train tfrecord dataset')
flags.DEFINE_integer('cnn_size',28,'The size we want to resize the number')
flags.DEFINE_integer('cnn_batch_size',1,'Size of the batch')
flags.DEFINE_float('cnn_learning_rate', 0.001, "")


def main(_argv):

    if FLAGS.mode == 'predict':

        # Detect where the license plate is on the image
        image = license_plate_detection(FLAGS.classes, FLAGS.yolo_weights, FLAGS.yolo_output, FLAGS.yolo_cropped_output, FLAGS.batch_detection, FLAGS.input_images, FLAGS.image, FLAGS.yolo_size, FLAGS.num_classes)
        
        # Segment the cropped image
        characters = segementation(FLAGS.yolo_cropped_output, FLAGS.seg_output, image, FLAGS.batch_detection)
        
        # Recognize the license plate numbers
        license_plate_numbers = []
        for character in characters:
            prediction = character_reco(character)
            license_plate_numbers.append(prediction)

        print('The license plate of the car is {}'.format(license_plate_numbers))

    elif FLAGS.mode == 'yolo_fine_tune':
        
        # Fine tune the yolo model with the tfrecord you have created
        yolo_train(FLAGS.yolo_train_dataset, FLAGS.yolo_val_dataset, FLAGS.yolo_pretrained_weights, FLAGS.classes, FLAGS.yolo_size, FLAGS.num_classes, FLAGS.yolo_batch_size, FLAGS.yolo_learning_rate, FLAGS.yolo_epochs)

    elif FLAGS.mode == 'yolo_batch_detect':

        # Detect the license plate position on multiple images
        license_plate_detection(FLAGS.classes, FLAGS.yolo_weights, FLAGS.yolo_output, FLAGS.yolo_cropped_output, True, FLAGS.input_images, FLAGS.image, FLAGS.yolo_size, FLAGS.num_classes)        

    elif FLAGS.mode == 'segment_only':

        # Segment and save the image
        # N.B. enable the batch_detection option if you want to save you sgemented
        # characters in the FLAGS.seg_output folder
        segementation(FLAGS.yolo_cropped_output, FLAGS.seg_output, FLAGS.image, FLAGS.batch_detection)

    elif FLAGS.mode == 'cnn_fine_tune':

        # Fine tune the cnn model with the tfrecord you created after labelising the output of the segmentation
        cnn_train(FLAGS.cnn_train_dataset, FLAGS.cnn_val_dataset, FLAGS.cnn_size, FLAGS.cnn_batch_size, FLAGS.cnn_learning_rate)

    # You can also modify the pre train model of the character reco by going to the character_reco 
    # folder and do `python pre_train.py`
if __name__ == "__main__":
    app.run(main)