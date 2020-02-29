import os   
import numpy as np 
import tensorflow as tf
import cv2
from absl import flags, logging, app
from absl.flags import FLAGS

from .datasets import dataset as dataset
# flags.DEFINE_string('dataset','./data/tfrecord/test.tfrecord','Path to the train tfrecord dataset')
# flags.DEFINE_integer('size',28,'The size we want to resize the number')
# flags.DEFINE_integer('batch_size',1,'Size of the batch')
# flags.DEFINE_float('learning_rate', 0.001, "")
# flags.DEFINE_integer('initial_step', 0, "")
# flags.DEFINE_integer('final_step', 5000, "")
# flags.DEFINE_integer('eval_freq', 1, "")
# flags.DEFINE_integer('epochs', 1,'number of epochs')

def main(train_dataset, val_dataset, size=28, batch_size=8, learning_rate=1e-3):

    model = tf.keras.models.load_model('./character_reco/models/CNN_saved_pre_trained_model')

    first_layer = model.get_layer('cnn_maxpool_1')
    dataset.freeze_all(first_layer)


    model.summary()

    train_dataset = dataset.load_tfrecord_dataset(train_dataset, size)
    train_dataset = train_dataset.shuffle(buffer_size=8)
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = dataset.load_tfrecord_dataset(val_dataset, size)
    val_dataset = train_dataset.shuffle(buffer_size=8)
    # val_dataset = train_dataset.batch(batch_size)

    # Create training operations

    optimizer = tf.optimizers.SGD(learning_rate)
    # optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_accuracy')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    @tf.function
    def forward(features, training=False):
        print('build eval')
        predictions = model.call(features[0], training=training)
        loss = tf.losses.categorical_crossentropy(
                y_true=features[1],
                y_pred=predictions)
        return loss, predictions

    @tf.function
    def train_step(features):
        print('build train')
        with tf.GradientTape() as tape:
            loss, predictions = forward(features, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    eval_freq = 10

    for step, features in enumerate(train_dataset):
        train_step(features)
        if step % eval_freq == 0:
            for train_features in train_dataset.take(10):
                # print(train_features[0].numpy()[0])
                loss, predictions = forward(train_features)
                # cv2.imshow('{}'.format(predictions[0]),train_features[0].numpy()[0])
                # cv2.waitKey()
                train_accuracy(tf.math.argmax(train_features[1], axis=-1), predictions)
                train_loss(loss)
            for test_features in val_dataset.take(10):
                loss, predictions = forward(test_features)
                test_accuracy(tf.math.argmax(test_features[1], axis=-1), predictions)
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

        img_raw = tf.image.decode_image(
            open('./character_reco/data/jpg/valid/truck_plate_8_0_char_7.jpg', 'rb').read(), channels=1)
        img = tf.expand_dims(img_raw, 0)

        img = dataset.transform_images(img, size)

        model.predict(img)

        tf.keras.models.save_model(model,'./models/{}_saved_final_model'.format('CNN'))
