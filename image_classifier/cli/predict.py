import tensorflow as tf
import tensorflow_hub as hub
import argparse

print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)

def predict():
    # args
    parser = argparse.ArgumentParser(
        description='Example with long option names',
    )

    parser.add_argument('image_path', action="store")
    parser.add_argument('model', action="store")
    parser.add_argument('--top_k', action="store", dest='top_k', type=int)

    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model
    top_k = args.top_k

    reloaded_keras_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    reloaded_keras_model.summary()

    print("image_path: ", image_path)
    print("top_k: ", top_k)

predict()
