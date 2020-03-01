import os
import sys
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from absl import app, flags, logging

from datasets import mnist
from models.cnn import CNN

from datasets.dataset import transform_images

import cv2

datasets = {
    'mnist': mnist
}

models = {
    'cnn': CNN
}


def main(argv):

    # Load dataset, model and optimizer
    dataset = datasets[FLAGS.dataset]
    train_dataset = dataset.load(FLAGS.batch_size, split='train')
    test_dataset = dataset.load(FLAGS.batch_size, split='test')

    model = models[FLAGS.model](ch=FLAGS.width_multiplier)
    model.build(input_shape=(FLAGS.batch_size, 28, 28, 1))

    optimizer = tf.optimizers.SGD(FLAGS.learning_rate)
    # optimizer = tf.optimizers.Adam(FLAGS.learning_rate)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_accuracy')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

   
    @tf.function
    def forward(features, training=False):
        print('build eval')
        predictions = model.call(features['image'], training=training)
        loss = tf.losses.categorical_crossentropy(
                y_true=features['label'],
                y_pred=predictions)
        return loss, predictions

    @tf.function
    def train_step(features):
        print('build train')
        with tf.GradientTape() as tape:
            loss, predictions = forward(features, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    
    # Restore checkpoint
    if FLAGS.restore: ckpt.restore(manager.latest_checkpoint)

# ================================ TRAINING ====================================
    
    for step, features in train_dataset.enumerate(FLAGS.initial_step):
        train_step(features)

        if step % FLAGS.eval_freq == 0:
            for train_features in train_dataset.take(10):
                loss, predictions = forward(train_features)
                train_accuracy(tf.math.argmax(train_features['label'], axis=-1), predictions)
                train_loss(loss)
            for test_features in test_dataset.take(10):
                loss, predictions = forward(test_features)
                test_accuracy(tf.math.argmax(test_features['label'], axis=-1), predictions)
                test_loss(loss)

            template = 'step: {:06d} - train loss/acc: {:3.2f}/{:2.2%} - test loss/acc: {:3.2f}/{:2.2%}'
            print(template.format(step, 
                train_loss.result(), train_accuracy.result(), 
                test_loss.result(), test_accuracy.result()))
            logging.info(template.format(step, 
                train_loss.result(), train_accuracy.result(), 
                test_loss.result(), test_accuracy.result()))
            
            # Reset the metrics for the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
        
        if step % FLAGS.save_freq == 0 and step != 0:
            manager.save()  
        
        ckpt.step.assign_add(1)
        if step == FLAGS.final_step + 1: break
    
    img_raw = tf.image.decode_image(
            open('../output.jpg', 'rb').read(), channels=1)
    img = tf.expand_dims(img_raw, 0)

    img = transform_images(img, 28)
    model.predict(img)
    
    tf.keras.models.save_model(model,'./character_reco/models/{}_saved_pre_trained_model'.format('CNN'))


if __name__ == '__main__':

    FLAGS = flags.FLAGS

    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs'), "")
    flags.DEFINE_string('experiment_name', 'test', "")

    flags.DEFINE_enum('dataset', 'mnist', ['mnist'], "")
    flags.DEFINE_enum('model', 'cnn', ['cnn'], "")
    flags.DEFINE_integer('width_multiplier', 64, "")

    flags.DEFINE_integer('initial_step', 0, "")
    flags.DEFINE_integer('final_step', 5000, "")
    flags.DEFINE_integer('save_freq', 1000, "")
    flags.DEFINE_integer('eval_freq', 1000, "")

    flags.DEFINE_bool('restore', False, "")
    flags.DEFINE_integer('batch_size', 128, "")
    flags.DEFINE_float('learning_rate', 0.001, "")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit(True)
    app.run(main)
