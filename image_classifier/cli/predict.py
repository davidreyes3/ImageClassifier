import tensorflow as tf
import tensorflow_hub as hub

import argparse

print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)


def predict():
    saved_keras_model_filepath = '../udacityIntroToMLImageClassification_1588776144.h5'
    reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath,
                                                      custom_objects={'KerasLayer': hub.KerasLayer})
    reloaded_keras_model.summary()


predict()
