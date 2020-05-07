import argparse
import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import logging
import os
import utils
import arguments

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# used:
# May 7, 2020
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# disables TensorFlow CPU message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict(image_path, model, top_k=1):
    if top_k < 1:
        top_k = 1
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = utils.process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)

    predictions = model.predict(processed_test_image)

    # get indices
    pred_sorted = np.argsort(predictions)
    pred_sorted_flipped = np.fliplr(pred_sorted)
    indecies_top_k = pred_sorted_flipped[:, :top_k]
    # used: May7, 2020 https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

    probs = np.take(predictions, indecies_top_k)
    classes = indecies_top_k + 1

    return probs[0, :], classes[0, :]


def run():
    arg_obj = arguments.ArgumentClass()

    class_names = None
    if arg_obj.classes_path is not None:
        with open(arg_obj.classes_path, 'r') as f:
            class_names = json.load(f)

    reloaded_keras_model = tf.keras.models.load_model(arg_obj.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # predict
    probs, classes = predict(arg_obj.image_path, reloaded_keras_model, arg_obj.top_k)

    # print probs with integer classes or with class_names
    for i, (prob, class_index) in enumerate(zip(probs, classes)):
        prob = prob * 100
        if class_names is not None:
            print("Flower name: {} - {:04.2f}%".format(class_names[str(class_index)], prob))
        else:
            print("Flower index: {} - {:04.2f}%".format(class_index, prob))


run()

