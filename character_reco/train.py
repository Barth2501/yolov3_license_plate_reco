import os   
import numpy as np 
import tensorflow as tf
import cv2
from absl import flags, logging, app
from absl.flags import FLAGS

import datasets.dataset as dataset

flags.DEFINE_string('dataset','./data/tfrecord/test.tfrecord','Path to the train tfrecord dataset')
flags.DEFINE_integer('size',28,'The size we want to resize the number')
flags.DEFINE_integer('batch_size',1,'Size of the batch')
flags.DEFINE_float('learning_rate', 0.001, "")
flags.DEFINE_integer('initial_step', 0, "")
flags.DEFINE_integer('final_step', 5000, "")
flags.DEFINE_integer('eval_freq', 1, "")
flags.DEFINE_integer('epochs', 1,'number of epochs')

def main(argv):

    model = tf.keras.models.load_model('./models/CNN_saved_pre_trained_model')

    first_layer = model.get_layer('cnn_conv_2')
    dataset.freeze_all(first_layer)
    second_layer = model.get_layer('cnn_dense')
    dataset.freeze_all(second_layer)

    model.summary()

    train_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=8)
    train_dataset = train_dataset.batch(FLAGS.batch_size)

    # Create training operations

    optimizer = tf.optimizers.SGD(FLAGS.learning_rate)
    # optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    ##TODO retirer la loss de la partie suivante car elle concerne seulemetn yolo
    
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
    
    for step, features in enumerate(train_dataset):
        train_step(features)
        if step % FLAGS.eval_freq == 0:
            for train_features in train_dataset.take(10):
                # print(train_features[0].numpy()[0])
                loss, predictions = forward(train_features)
                # cv2.imshow('{}'.format(predictions[0]),train_features[0].numpy()[0])
                # cv2.waitKey()
                train_accuracy(tf.math.argmax(train_features[1], axis=-1), predictions)
                train_loss(loss)
            # for test_features in test_dataset.take(10):
            #     loss, predictions = forward(test_features)
            #     test_accuracy(tf.math.argmax(test_features['label'], axis=-1), predictions)
            #     test_loss(loss)

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
        if step == FLAGS.final_step + 1: break
        img_raw = tf.image.decode_image(
            open('./data/jpg/output_output_cropped0.jpg', 'rb').read(), channels=1)
        img = tf.expand_dims(img_raw, 0)

        img = dataset.transform_images(img, FLAGS.size)

        model.predict(img)

        tf.keras.models.save_model(model,'./models/{}_saved_final_model'.format('CNN'))
        
    
    # avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    # avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    # for epoch in range(1, FLAGS.epochs + 1):
    #     for batch, (images, labels) in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             outputs = model(images, training=True)
    #             regularization_loss = tf.reduce_sum(model.losses)
    #             pred_loss = []
    #             for output, label in zip(outputs, labels):
    #                 pred_loss.append(avg_loss(label, output))
    #             total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    #         grads = tape.gradient(total_loss, model.trainable_variables)
    #         optimizer.apply_gradients(
    #             zip(grads, model.trainable_variables))
    #         logging.info("{}_train_{}, {}, {}".format(
    #             epoch, batch, total_loss.numpy(),
    #             list(map(lambda x: np.sum(x.numpy()), pred_loss))))
    #         avg_loss.update_state(total_loss)
    #     logging.info("{}, train: {}, val: {}".format(
    #         epoch,
    #         avg_loss.result().numpy(),
    #         avg_val_loss.result().numpy()))
    #     avg_loss.reset_states()
    #     avg_val_loss.reset_states()
    #     model.save_weights(
    #             'checkpoints/cnn_train_{}.tf'.format(epoch))


if __name__=='__main__':
    app.run(main)
    