from typing import List, Tuple
import tensorflow as tf
import numpy as np
from PIL import Image

def pred_class(model: tf.keras.Model,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224),
               ):

    # Open image
    img = image

    # Create transformation for image (if one doesn't exist)
    def preprocess_image(image):
        image = tf.image.resize(image, image_size)
        image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        return image

    # Predict on image
    img_array = np.array(img)
    img_array = preprocess_image(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)

    # Convert prediction probabilities -> prediction labels
    predicted_label = np.argmax(predictions, axis=1)
    classname = class_names[predicted_label[0]]
    prob = predictions

    return prob
