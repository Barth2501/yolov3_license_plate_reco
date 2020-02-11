import os   
import numpy as np 
import tensorflow as tf

from operator import itemgetter
from absl import flags, logging, app
from absl.flags import FLAGS
from  tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Sequential

from models import (
    YoloV3, YoloLoss,
    yolo_anchors, yolo_anchor_masks
)
import dataset as dataset

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('classes', './data/plaques.numbers', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_string('output_dir', os.path.join('..', 'outputs'), "")
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('epochs', 2, 'number of epochs')


def main(argv):
    
    # Create working directories
    # experiment_dir  = os.path.join(FLAGS.output_dir,
    #     FLAGS.experiment_name, FLAGS.model, FLAGS.dataset)
    
    # checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    # saved_model_dir = os.path.join(experiment_dir, 'saved_models')
    # os.makedirs(checkpoints_dir, exist_ok=True)
    # os.makedirs(saved_model_dir, exist_ok=True)

    # Logging training informations
    # logging.get_absl_handler().use_absl_log_file('logs', experiment_dir)

    # Load Model
    model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
    model.load_weights(FLAGS.weights)
    # freeze only the first layer
    darknet = model.get_layer('yolo_darknet')
    freeze_all(darknet)
    # # freeze every layers
    # freeze_all(model)
    model.summary()

    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # ============================== Read data =================================
    # see https://www.tensorflow.org/datasets/splits

    train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # Create training operations
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    for epoch in range(1, FLAGS.epochs + 1):
        for batch, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
            logging.info("{}_train_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_loss.update_state(total_loss)
        for batch, (images, labels) in enumerate(val_dataset):
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss
            logging.info("{}_val_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_val_loss.update_state(total_loss)
        logging.info("{}, train: {}, val: {}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy()))
        avg_loss.reset_states()
        avg_val_loss.reset_states()
        model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))

if __name__=='__main__':
    app.run(main)