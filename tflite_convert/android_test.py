import tensorflow as tf
from absl import flags, logging, app
import os

def main(_argv):

  if FLAGS.model == 'yolo':
    # Convert the model.
    model = tf.saved_model.load('../models/YOLO_saved_final_model/')
    concrete_func = model.signatures[
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, 416, 416, 3])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    ## Attention le fichier tflite peut ne pas etre géré par andorid studio a cause du allow_custom_ops 
    ## Voir un autre workaround si cela ne marche pas
    converter.allow_custom_ops = True

    tflite_model = converter.convert()

    open("tflite_models/yolo_model.tflite", "wb").write(tflite_model)
    print('The model have been saved in {}/tflite_models/ as yolo_model.tflite'.format(os.getcwd()))
  
  elif FLAGS.model == 'cnn':
    # Convert the model.
    model = tf.saved_model.load('../models/CNN_saved_final_model/')
    concrete_func = model.signatures[
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, 28, 28, 1])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    open("tflite_models/cnn_model.tflite", "wb").write(tflite_model)
    print('The model have been saved in {}/tflite_models/ as cnn_model.tflite'.format(os.getcwd()))

  elif FLAGS.model == 'ssd':
  # Convert the model.
    model = tf.saved_model.load('../models/SSD_saved_final_model/')
    concrete_func = model.signatures[
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, 300, 300, 3])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    open("tflite_models/cnn_model.tflite", "wb").write(tflite_model)
    print('The model have been saved in {}/tflite_models/ as ssd_model.tflite'.format(os.getcwd()))    


if __name__=='__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('model','yolo','Choose the model you want to convert to tflite')
    app.run(main)
