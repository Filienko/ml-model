import numpy as np
import tensorflow as tf

# Settings
img_height = 32
img_width = 32
model = tf.keras.models.load_model('image_model')
img_path = "1.png"
class_names = ["Male","Female"]

image = tf.keras.utils.load_img(img_path)
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = tf.image.resize(input_arr, (img_height, img_width))
predictions = model.predict(input_arr)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))