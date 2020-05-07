import tensorflow as tf


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, size=(224, 224))
    image /= 255
    return image.numpy()